"""
Static pool-heater delivery model, v2.

Decomposition (memoryless, all dynamics live in the pool RC model):

    E[Q_in(k)]  =  duty(k) * level(k)

    duty(k)  = sigmoid( a0 + a1*(T_sp - T_pool) + a2*session_active )
        - probability the heater is "on" in this 10-min interval
        - sigmoid keeps it in [0,1], smooth and differentiable for MPC

    level(k) = max( L0 + L1*max(0, T_out_ref - T_out) - L2*(T_pool - T_pool_ref),
                    L_min )
        - kW delivered when on, driven by the DH heating curve (proxied by T_out)
          and the achievable secondary dT (proxied by T_pool)

Inputs at MPC time: T_sp [C], T_pool [C], T_out [C], session_active [0/1]
Output: Q_in [W]

Why this shape (informed by data diagnostics):
  - corr(Q_in_when_on, T_supply) = 0.90, corr(Q_in_when_on, T_out) = -0.78
    => when on, level is set by the heating curve, not by the pool error
  - P(on) rises monotonically from 0.16 (1 K above setpoint) to 1.00 (1+ K below)
    => setpoint acts as a duty-cycle gate, not a magnitude controller
  - swim sessions cause a large mid-day Q drop on session days
    => session_active belongs in the duty term, not the level term
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ---------- config ----------
CSV_PATH      = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\Pool_heater_with_weather_10min_.csv"
LOG_PATH      = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\ON_OFF pool heater.txt"
SESSION_PATH  = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\holding_tank_sessions_full_daily_log.csv"
# FIX: Wrap the string in Path() so it has the .mkdir() method
OUT_DIR       = Path(r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CP_WATER     = 4186.0
RHO_WATER    = 1000.0
WARMUP_MIN   = 20
HELD_OUT_IDS = [3, 7]
SAMPLE_DT_S  = 600
SESSION_PAD_MIN = 10   # treat ±10 min around a session as "active" (cover off / interlocks)


# ---------- data loaders ----------
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    m_dot = df["flow_lph"] * RHO_WATER / 3600.0 / 1000.0
    df["Q_in_W"] = (m_dot * CP_WATER * df["diff_temperature_K"]).clip(lower=0)
    df = df.rename(columns={
        "Outside air temp from station": "T_out",
        "pool temperature": "T_pool",
        "temp_t1_C": "T_return",
        "temp_t2_C": "T_supply",
    })
    return df


def parse_setpoint_log(path: Path, year: int = 2026) -> pd.DataFrame:
    months = {f"{i:02d}": i for i in range(1, 13)}
    line_re = re.compile(
        r"(\d{2})/(\d{2}).*?(\d{1,2}):(\d{2}).*?"
        r"(?:from\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)|to\s*(\d+(?:\.\d+)?))",
        re.IGNORECASE,
    )
    rows = []
    for raw in open(path):
        m = line_re.search(raw.strip())
        if not m:
            continue
        dd, mm, hh, mi = m.group(1), m.group(2), m.group(3), m.group(4)
        from_v, to_v = m.group(5), (m.group(6) or m.group(7))
        rows.append({
            "timestamp": pd.Timestamp(year=year, month=months[mm], day=int(dd),
                                       hour=int(hh), minute=int(mi)),
            "T_sp_from": float(from_v) if from_v else np.nan,
            "T_sp_new":  float(to_v),
        })
    log = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    log.attrs["initial_T_sp"] = (log["T_sp_from"].dropna().iloc[0]
                                  if log["T_sp_from"].notna().any()
                                  else log["T_sp_new"].iloc[0])
    return log


def parse_sessions(path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return list of (start, end) timestamps from the daily session log."""
    df = pd.read_csv(path)
    intervals = []
    for _, row in df.iterrows():
        if row["n_sessions"] == 0 or "No detected" in str(row["sessions"]):
            continue
        date = pd.Timestamp(row["date"])
        for chunk in str(row["sessions"]).split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            t0_s, t1_s = chunk.split("-")
            t0 = date + pd.Timedelta(hours=int(t0_s.split(":")[0]),
                                      minutes=int(t0_s.split(":")[1]))
            t1 = date + pd.Timedelta(hours=int(t1_s.split(":")[0]),
                                      minutes=int(t1_s.split(":")[1]))
            intervals.append((t0, t1))
    return intervals


def build_setpoint_series(df: pd.DataFrame, log: pd.DataFrame) -> pd.Series:
    sp = pd.Series(np.nan, index=df["timestamp"], name="T_sp")
    sp.iloc[0] = log.attrs["initial_T_sp"]
    for _, row in log.iterrows():
        idx = sp.index.searchsorted(row["timestamp"], side="left")
        if idx < len(sp):
            sp.iloc[idx] = row["T_sp_new"]
    return sp.ffill().reset_index(drop=True)


def build_session_flag(df: pd.DataFrame,
                       intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
                       pad_min: int = SESSION_PAD_MIN) -> pd.Series:
    flag = pd.Series(0, index=df.index, dtype=int)
    pad = pd.Timedelta(minutes=pad_min)
    ts = df["timestamp"]
    for t0, t1 in intervals:
        m = (ts >= t0 - pad) & (ts <= t1 + pad)
        flag[m] = 1
    return flag


def segment_events(df: pd.DataFrame, log: pd.DataFrame,
                   warmup_min: int = WARMUP_MIN) -> pd.DataFrame:
    df = df.copy()
    df["event_id"] = -1
    df["in_warmup"] = False
    bounds = list(log["timestamp"]) + [df["timestamp"].iloc[-1] + pd.Timedelta(seconds=1)]
    for eid, (t0, t1) in enumerate(zip(bounds[:-1], bounds[1:])):
        m = (df["timestamp"] >= t0) & (df["timestamp"] < t1)
        df.loc[m, "event_id"] = eid
        warm_end = t0 + pd.Timedelta(minutes=warmup_min)
        df.loc[m & (df["timestamp"] < warm_end), "in_warmup"] = True
    return df


# ---------- model ----------
def _sigmoid(x):
    # numerically stable
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def predict_Q(params, T_sp, T_pool, T_out, session):
    """
    params = [a0, a1, a2,             # duty: bias, error gain, session shift
              L0, L1, L2,              # level: base, T_out gain, T_pool penalty
              T_out_ref, T_pool_ref,
              L_min]
    """
    a0, a1, a2, L0, L1, L2, T_out_ref, T_pool_ref, L_min = params
    err  = T_sp - T_pool
    duty = _sigmoid(a0 + a1 * err + a2 * session)
    level = (L0
             + L1 * np.maximum(0.0, T_out_ref - T_out)
             - L2 * (T_pool - T_pool_ref))
    level = np.maximum(level, L_min)
    return duty * level


def fit_model(df_fit: pd.DataFrame):
    args = (df_fit["T_sp"].values,
            df_fit["T_pool"].values,
            df_fit["T_out"].values,
            df_fit["session"].values.astype(float))
    Q_meas = df_fit["Q_in_W"].values

    # Initial guesses
    p0 = [ 0.0,   2.0, -2.0,
          18000.0, 300.0, 500.0,
          5.0, 32.0,
          5000.0]
    lb = [-5.0, 0.0, -10.0,
           5000.0,    0.0,    0.0,
          -10.0, 25.0,
              0.0]
    ub = [ 5.0, 10.0,   2.0,
          40000.0, 2000.0, 3000.0,
            20.0, 40.0,
          15000.0]

    def residuals(p):
        return predict_Q(p, *args) - Q_meas

    return least_squares(residuals, p0, bounds=(lb, ub),
                         method="trf", loss="soft_l1", f_scale=2000.0,
                         max_nfev=10000)


# ---------- metrics ----------
def metrics(Q_true, Q_pred, dt_s=SAMPLE_DT_S):
    err = Q_pred - Q_true
    e_true = np.sum(Q_true) * dt_s / 3.6e6
    e_pred = np.sum(Q_pred) * dt_s / 3.6e6
    return {
        "MAE_W":   float(np.mean(np.abs(err))),
        "RMSE_W":  float(np.sqrt(np.mean(err**2))),
        "E_true_kWh": float(e_true),
        "E_pred_kWh": float(e_pred),
        "E_err_pct":  float(100.0 * (e_pred - e_true) / e_true) if e_true != 0 else float("nan"),
        "n":       int(len(Q_true)),
    }


# ---------- main ----------
def main():
    df  = load_csv(CSV_PATH)
    log = parse_setpoint_log(LOG_PATH)
    sessions = parse_sessions(SESSION_PATH)
    print(f"Parsed {len(log)} setpoint changes, {len(sessions)} swim sessions")

    df["T_sp"]    = build_setpoint_series(df, log)
    df["session"] = build_session_flag(df, sessions)
    df = segment_events(df, log)

    print(f"Session flag set on {df['session'].sum()} of {len(df)} samples "
          f"({100*df['session'].mean():.1f}%)")

    fit_pool = df.dropna(subset=["T_pool", "T_out", "T_sp", "Q_in_W"]).copy()
    fit_pool = fit_pool[(fit_pool["event_id"] >= 0) & (~fit_pool["in_warmup"])]

    train_mask = ~fit_pool["event_id"].isin(HELD_OUT_IDS)
    df_train, df_test = fit_pool[train_mask], fit_pool[~train_mask]
    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)} "
          f"(held out events: {HELD_OUT_IDS})")

    res = fit_model(df_train)
    p = res.x
    names = ["a0_bias", "a1_err_per_K", "a2_session",
             "L0_W", "L1_W_per_K_out", "L2_W_per_K_pool",
             "T_out_ref_C", "T_pool_ref_C", "L_min_W"]
    fitted = dict(zip(names, [float(v) for v in p]))
    print("\nFitted parameters:")
    for k, v in fitted.items():
        print(f"  {k:18s} = {v:12.4f}")

    Q_train_pred = predict_Q(p, df_train["T_sp"].values,
                                df_train["T_pool"].values,
                                df_train["T_out"].values,
                                df_train["session"].values.astype(float))
    Q_test_pred  = predict_Q(p, df_test["T_sp"].values,
                                df_test["T_pool"].values,
                                df_test["T_out"].values,
                                df_test["session"].values.astype(float))

    train_metrics = metrics(df_train["Q_in_W"].values, Q_train_pred)
    test_metrics  = metrics(df_test["Q_in_W"].values,  Q_test_pred)
    print("\nTrain:", {k: round(v, 2) for k, v in train_metrics.items()})
    print("Test: ", {k: round(v, 2) for k, v in test_metrics.items()})

    print("\nPer-held-out-event integrated energy:")
    per_event = []
    for eid in HELD_OUT_IDS:
        ev = df_test[df_test["event_id"] == eid]
        if len(ev) == 0:
            continue
        Qp = predict_Q(p, ev["T_sp"].values, ev["T_pool"].values,
                          ev["T_out"].values, ev["session"].values.astype(float))
        m = metrics(ev["Q_in_W"].values, Qp)
        m["event_id"] = int(eid)
        m["t_start"]  = str(ev["timestamp"].iloc[0])
        m["t_end"]    = str(ev["timestamp"].iloc[-1])
        per_event.append(m)
        print(f"  event {eid}: n={m['n']:3d}  E_true={m['E_true_kWh']:6.2f} kWh  "
              f"E_pred={m['E_pred_kWh']:6.2f} kWh  err={m['E_err_pct']:+5.1f}%  "
              f"MAE={m['MAE_W']:6.0f} W")

    with open(OUT_DIR / "fitted_params_v2.json", "w") as f:
        json.dump({
            "params": fitted,
            "train_metrics": train_metrics,
            "test_metrics":  test_metrics,
            "per_event_metrics": per_event,
            "held_out_event_ids": HELD_OUT_IDS,
            "warmup_minutes": WARMUP_MIN,
            "session_pad_minutes": SESSION_PAD_MIN,
        }, f, indent=2)

    make_plots(df, fit_pool, p, train_metrics, test_metrics)
    write_mpc_callable(fitted)
    return df, fit_pool, p, fitted


def make_plots(df_all, df_fit, p, train_metrics, test_metrics):
    err = df_fit["T_sp"] - df_fit["T_pool"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, mask, title in [
        (axes[0], df_fit["session"] == 0, "session = 0"),
        (axes[1], df_fit["session"] == 1, "session = 1"),
    ]:
        sub = df_fit[mask]
        sc = ax.scatter(err[mask], sub["Q_in_W"] / 1000.0,
                        c=sub["T_out"], cmap="coolwarm_r", s=8, alpha=0.6)
        if len(sub) > 0:
            Tp = float(np.median(sub["T_pool"]))
            To = float(np.median(sub["T_out"]))
            e_grid = np.linspace(-2.5, 2.5, 200)
            Qc = predict_Q(p, Tp + e_grid, np.full_like(e_grid, Tp),
                              np.full_like(e_grid, To),
                              np.full_like(e_grid, 0 if title.endswith("0") else 1))
            ax.plot(e_grid, Qc / 1000.0, "k-", lw=2,
                    label=f"model @ T_pool={Tp:.1f}, T_out={To:.1f}")
            ax.legend(loc="upper left", fontsize=8)
        plt.colorbar(sc, ax=ax, label="T_out [C]")
        ax.set_xlabel("T_sp - T_pool [K]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Q_in [kW]")
    fig.suptitle("Static delivery model: data vs fit, split by swim session")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fit_vs_error_v2.png", dpi=130)
    plt.close(fig)

    Qp_all = predict_Q(p, df_fit["T_sp"].values, df_fit["T_pool"].values,
                          df_fit["T_out"].values,
                          df_fit["session"].values.astype(float))
    test_mask = df_fit["event_id"].isin(HELD_OUT_IDS).values
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_fit["Q_in_W"].values[~test_mask] / 1000.0,
               Qp_all[~test_mask] / 1000.0,
               s=8, alpha=0.4, label=f"train (n={(~test_mask).sum()})")
    ax.scatter(df_fit["Q_in_W"].values[test_mask] / 1000.0,
               Qp_all[test_mask] / 1000.0,
               s=14, alpha=0.7, color="C3", label=f"held-out (n={test_mask.sum()})")
    lim = max(df_fit["Q_in_W"].max(), Qp_all.max()) / 1000.0 * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Measured Q_in [kW]")
    ax.set_ylabel("Predicted Q_in [kW]")
    ax.set_title(
        f"Parity v2 — train MAE={train_metrics['MAE_W']/1000:.2f} kW, "
        f"test MAE={test_metrics['MAE_W']/1000:.2f} kW, "
        f"test E err={test_metrics['E_err_pct']:+.1f}%"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "parity_v2.png", dpi=130)
    plt.close(fig)

    df_plot = df_all.dropna(subset=["T_pool"]).copy()
    Qp_full = predict_Q(p, df_plot["T_sp"].values, df_plot["T_pool"].values,
                           df_plot["T_out"].values,
                           df_plot["session"].values.astype(float))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df_plot["timestamp"], df_plot["Q_in_W"] / 1000.0,
                 label="measured", color="k", lw=0.9)
    axes[0].plot(df_plot["timestamp"], Qp_full / 1000.0,
                 label="model", color="C3", lw=1.0, alpha=0.85)
    axes[0].set_ylabel("Q_in [kW]"); axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3)

    axes[1].plot(df_plot["timestamp"], df_plot["T_pool"], label="T_pool", color="C0")
    axes[1].plot(df_plot["timestamp"], df_plot["T_sp"],   label="T_sp",
                 color="C1", drawstyle="steps-post")
    axes[1].set_ylabel("Temperature [C]"); axes[1].legend(loc="upper right"); axes[1].grid(alpha=0.3)

    axes[2].plot(df_plot["timestamp"], df_plot["T_out"], label="T_outside", color="C2")
    axes[2].set_ylabel("T_outside [C]"); axes[2].set_xlabel("time")
    axes[2].legend(loc="upper right"); axes[2].grid(alpha=0.3)

    in_session = df_all["session"].values
    ts = df_all["timestamp"].values
    starts, ends = [], []
    i = 0
    while i < len(in_session):
        if in_session[i]:
            j = i
            while j < len(in_session) and in_session[j]:
                j += 1
            starts.append(ts[i]); ends.append(ts[j - 1])
            i = j
        else:
            i += 1
    for s, e in zip(starts, ends):
        for ax in axes:
            ax.axvspan(s, e, color="C0", alpha=0.10)
    for eid in HELD_OUT_IDS:
        ev = df_all[df_all["event_id"] == eid]
        if len(ev) == 0:
            continue
        for ax in axes:
            ax.axvspan(ev["timestamp"].iloc[0], ev["timestamp"].iloc[-1],
                       color="red", alpha=0.10)
    fig.suptitle("v2: pale blue = swim sessions, red = held-out DR events")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "timeseries_v2.png", dpi=130)
    plt.close(fig)


def write_mpc_callable(fitted: dict):
    code = f'''"""
Static pool-heater delivery model (v2).
Inputs: T_sp [C], T_pool [C], T_out [C], session_active [0/1]
Output: Q_in [W]
Memoryless. All thermal dynamics belong in the pool RC model.
"""
import numpy as np

PARAMS = {json.dumps(fitted, indent=4)}

def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def level(T_pool, T_out):
    p = PARAMS
    L = (p["L0_W"]
         + p["L1_W_per_K_out"]  * np.maximum(0.0, p["T_out_ref_C"] - T_out)
         - p["L2_W_per_K_pool"] * (T_pool - p["T_pool_ref_C"]))
    return np.maximum(L, p["L_min_W"])

def duty(T_sp, T_pool, session_active):
    p = PARAMS
    err = np.asarray(T_sp, dtype=float) - np.asarray(T_pool, dtype=float)
    return _sigmoid(p["a0_bias"]
                    + p["a1_err_per_K"] * err
                    + p["a2_session"]   * np.asarray(session_active, dtype=float))

def predict_Q_in(T_sp, T_pool, T_out, session_active=0):
    return duty(T_sp, T_pool, session_active) * level(T_pool, T_out)
'''
    (OUT_DIR / "pool_heater_delivery_model_v2.py").write_text(code)


if __name__ == "__main__":
    main()