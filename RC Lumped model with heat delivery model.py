"""
pool_1r1c_coupled_layer1_lumpedR_compare_water_onlyC.py
=======================================================

Compare three coupled day-ahead pool models:

1) Explicit evaporation + single R
2) Lumped losses only + single R
3) Lumped losses only + separate R_unocc and R_occ

Layer 1 heater model:
    - Hysteresis ON/OFF logic
    - P_on(T_out, max(T_sp - T_pool, 0))

Validation:
    - chronological split
    - 70% calibration / 30% validation
    - 24 h open-loop forecast on last validation block

UPDATED:
    - C_pool uses WATER ONLY
    - removed +20% concrete shell addition
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS — EDIT PATHS
# ══════════════════════════════════════════════════════════════════════════════
HEATER_CSV    = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\Pool_heater_with_weather_10min_.csv"
SENSOR_XLSX   = r"M:\PhD\03 Experiments\Dalgård\Raw data\AHU\EXPORT_20-02-2026_06-03-2026.xlsx"
OCCUPANCY_CSV = r"M:\PhD\03 Experiments\Dalgård\holding_tank_date_by_date_occupancy_log.csv"

# ══════════════════════════════════════════════════════════════════════════════
# FIXED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
KH       = 4.005e-8
AF_POOL  = 72.0
LV       = 2_454_000
DT       = 600.0

# WATER-ONLY THERMAL CAPACITANCE
C_WATER  = 1000 * 4186 * 90.0   # J/K
C_FIXED  = C_WATER              # water only, no extra concrete shell

AF_UNOCC = 0.15
AF_OCC   = 1.0

BLOCK       = 144
TRAIN_RATIO = 0.70

FLOW_ON_THRESHOLD_LPH = 100.0
DELTA_C               = 0.66
INITIAL_SETPOINT_C    = 31.5
MIN_OFF_DURATION_MIN  = 30
MIN_ON_DURATION_MIN   = 20

SETPOINT_EVENTS = [
    ("2026-02-18 15:00:00", 33.5),
    ("2026-02-19 12:10:00", 31.5),
    ("2026-02-20 13:33:00", 33.5),
    ("2026-02-23 10:49:00", 31.5),
    ("2026-02-26 14:32:00", 33.5),
    ("2026-02-27 08:10:00", 31.5),
    ("2026-02-27 14:31:00", 33.5),
    ("2026-03-02 08:02:00", 31.5),
    ("2026-03-02 14:31:00", 33.5),
    ("2026-03-03 08:07:00", 31.5),
    ("2026-03-03 12:15:00", 33.5),
    ("2026-03-03 14:30:00", 31.5),
    ("2026-03-04 08:31:00", 33.5),
    ("2026-03-05 08:19:00", 31.5),
    ("2026-03-05 14:55:00", 33.5),
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def psat_magnus(T_C):
    return 610.94 * np.exp(17.625 * T_C / (243.04 + T_C))

def Q_evap_W(T_pool, T_air, RH_pct, AF):
    Pw = psat_magnus(T_pool)
    Pa = (RH_pct / 100.0) * psat_magnus(T_air)
    dP = np.maximum(0.0, Pw - Pa)
    return KH * AF_POOL * dP * AF * LV

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    res = y_pred - y_true
    mae = np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res ** 2))
    mbe = np.mean(res)
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r, _ = pearsonr(y_true, y_pred)
    cv_rmse = rmse / np.mean(y_true) * 100
    return dict(MAE=mae, RMSE=rmse, MBE=mbe, R2=r2, R=r, CV_RMSE=cv_rmse)

def fill_short_runs(binary_array, target_value, max_run_len):
    arr = np.asarray(binary_array, dtype=int).copy()
    n = len(arr)
    i = 0
    while i < n:
        if arr[i] == target_value:
            start = i
            while i < n and arr[i] == target_value:
                i += 1
            end = i - 1
            if (end - start + 1) <= max_run_len:
                arr[start:end + 1] = 1 - target_value
        else:
            i += 1
    return arr

def filter_observed_state(binary_array, min_off_steps=0, min_on_steps=0):
    arr = np.asarray(binary_array, dtype=int).copy()
    if min_off_steps > 0:
        arr = fill_short_runs(arr, target_value=0, max_run_len=min_off_steps)
    if min_on_steps > 0:
        arr = fill_short_runs(arr, target_value=1, max_run_len=min_on_steps)
    return arr

def heater_state_hysteresis(T_pool, T_sp, prev_state, delta=DELTA_C):
    if T_pool < (T_sp - delta):
        return 1
    elif T_pool > (T_sp + delta):
        return 0
    else:
        return prev_state

def build_sensor_columns(raw):
    h0 = raw.iloc[0].fillna("")
    h1 = raw.iloc[1].fillna("")
    h2 = raw.iloc[2].fillna("")
    cols = []
    for a, b, c in zip(h0, h1, h2):
        parts = [str(x).strip() for x in [a, b, c] if str(x).strip() not in ["", "nan"]]
        cols.append("|".join(parts))
    return cols

def find_col(columns, contains_all):
    for c in columns:
        if all(s.lower() in c.lower() for s in contains_all):
            return c
    return None

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    # Heater CSV
    heater = pd.read_csv(HEATER_CSV)
    heater["timestamp"] = pd.to_datetime(heater["timestamp"], errors="coerce")
    heater = heater.sort_values("timestamp").set_index("timestamp")

    for c in heater.columns:
        heater[c] = pd.to_numeric(heater[c], errors="coerce")

    heater["Q_in_W_meas"] = heater["power_W"]
    heater["T_out"] = heater["Outside air temp from station"]

    heater["heater_on_obs_raw"] = (heater["flow_lph"] > FLOW_ON_THRESHOLD_LPH).astype(int)
    min_off_steps = int(np.ceil(MIN_OFF_DURATION_MIN / 10))
    min_on_steps  = int(np.ceil(MIN_ON_DURATION_MIN / 10))
    heater["heater_on_obs"] = filter_observed_state(
        heater["heater_on_obs_raw"].values,
        min_off_steps=min_off_steps,
        min_on_steps=min_on_steps
    )

    # Sensor export
    raw0 = pd.read_excel(SENSOR_XLSX, sheet_name="Export", header=None)
    cols = build_sensor_columns(raw0)

    raw = raw0.iloc[4:].copy()
    raw.columns = cols

    dt = pd.to_datetime(
        raw.iloc[:, 0].astype(str).str.strip() + " " + raw.iloc[:, 1].astype(str).str.strip(),
        errors="coerce"
    )

    sensor = pd.DataFrame(index=dt)
    sensor.index.name = "timestamp"

    t2_col     = find_col(raw.columns, ["T2", "Temperature"])
    rh3_rh_col = find_col(raw.columns, ["RH3", "Humidity"])
    rh3_t_col  = find_col(raw.columns, ["RH3", "Temperature"])

    sensor["T_pool"]  = pd.to_numeric(raw[t2_col], errors="coerce").values
    sensor["RH_pct"]  = pd.to_numeric(raw[rh3_rh_col], errors="coerce").values
    sensor["T_air_C"] = pd.to_numeric(raw[rh3_t_col], errors="coerce").values

    sensor = sensor.dropna(how="all").resample("10min").mean()
    sensor[["T_pool", "RH_pct", "T_air_C"]] = sensor[["T_pool", "RH_pct", "T_air_C"]].ffill()

    data = heater[["Q_in_W_meas", "T_out", "heater_on_obs"]].join(sensor, how="inner")
    data = data.dropna(subset=["Q_in_W_meas", "T_out", "T_pool", "RH_pct", "T_air_C"])

    # Occupancy
    occ_raw = pd.read_csv(OCCUPANCY_CSV)
    occ_mask = pd.Series(False, index=data.index)

    for _, row in occ_raw[occ_raw["occupancy_detected"] == "Yes"].iterrows():
        d = pd.to_datetime(row["date"], format="%d-%m-%Y").date()
        sh, sm = map(int, row["start_time"].split(":"))
        eh, em = map(int, row["end_time"].split(":"))
        sdt = pd.Timestamp(d.year, d.month, d.day, sh, sm)
        edt = pd.Timestamp(d.year, d.month, d.day, eh, em)
        occ_mask |= (data.index >= sdt) & (data.index <= edt)

    data["is_occ"] = occ_mask.astype(int)

    # Setpoint schedule
    data["T_sp"] = INITIAL_SETPOINT_C
    for ts, val in SETPOINT_EVENTS:
        data.loc[data.index >= pd.Timestamp(ts), "T_sp"] = val

    data["temp_error"] = data["T_sp"] - data["T_pool"]
    data["temp_error_pos"] = data["temp_error"].clip(lower=0)

    return data

# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 HEATER MODEL
# ══════════════════════════════════════════════════════════════════════════════
def fit_layer1_heater_model(data_cal):
    d_on = data_cal[data_cal["heater_on_obs"] == 1].copy()

    X = d_on[["T_out", "temp_error_pos"]].values
    y = d_on["Q_in_W_meas"].values / 1000.0

    model = LinearRegression()
    model.fit(X, y)

    return {
        "pon_coef": model.coef_,
        "pon_intercept": model.intercept_,
        "pon_min_kW": max(0.0, np.quantile(y, 0.02)),
        "pon_max_kW": np.quantile(y, 0.98),
    }

def predict_Pon_W(T_out, temp_error_pos, heater_params):
    q_kw = (
        heater_params["pon_intercept"]
        + heater_params["pon_coef"][0] * T_out
        + heater_params["pon_coef"][1] * temp_error_pos
    )
    q_kw = np.clip(q_kw, heater_params["pon_min_kW"], heater_params["pon_max_kW"])
    return q_kw * 1000.0

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATORS
# ══════════════════════════════════════════════════════════════════════════════
def simulate_explicit(T0, T_sp_series, T_air, RH_pct, T_out, params, heater_params, occ_flag, init_state):
    C = params["C_pool"]
    R = params["R_pool"]
    N = len(T_sp_series)

    T_pred = np.empty(N + 1)
    Qin = np.empty(N)
    Qe = np.empty(N)
    Ql = np.empty(N)
    state = np.empty(N, dtype=int)

    T_pred[0] = T0
    prev_state = int(init_state)

    for k in range(N):
        Tpk = T_pred[k]
        Tsp = T_sp_series[k]

        st = heater_state_hysteresis(Tpk, Tsp, prev_state)
        state[k] = st
        prev_state = st

        Qin[k] = predict_Pon_W(T_out[k], max(Tsp - Tpk, 0.0), heater_params) if st == 1 else 0.0

        AF = AF_OCC if occ_flag[k] else AF_UNOCC
        Qe[k] = Q_evap_W(Tpk, T_air[k], RH_pct[k], AF)
        Ql[k] = (Tpk - T_out[k]) / R

        T_pred[k + 1] = Tpk + (DT / C) * (Qin[k] - Ql[k] - Qe[k])

    return T_pred, Qin, Qe, Ql, state

def simulate_lumped_singleR(T0, T_sp_series, T_out, params, heater_params, init_state):
    C = params["C_pool"]
    R = params["R_pool"]
    N = len(T_sp_series)

    T_pred = np.empty(N + 1)
    Qin = np.empty(N)
    Ql = np.empty(N)
    state = np.empty(N, dtype=int)

    T_pred[0] = T0
    prev_state = int(init_state)

    for k in range(N):
        Tpk = T_pred[k]
        Tsp = T_sp_series[k]

        st = heater_state_hysteresis(Tpk, Tsp, prev_state)
        state[k] = st
        prev_state = st

        Qin[k] = predict_Pon_W(T_out[k], max(Tsp - Tpk, 0.0), heater_params) if st == 1 else 0.0
        Ql[k] = (Tpk - T_out[k]) / R

        T_pred[k + 1] = Tpk + (DT / C) * (Qin[k] - Ql[k])

    return T_pred, Qin, Ql, state

def simulate_lumped_twoR(T0, T_sp_series, T_out, occ_flag, params, heater_params, init_state):
    C = params["C_pool"]
    R_unocc = params["R_unocc"]
    R_occ   = params["R_occ"]
    N = len(T_sp_series)

    T_pred = np.empty(N + 1)
    Qin = np.empty(N)
    Ql = np.empty(N)
    state = np.empty(N, dtype=int)

    T_pred[0] = T0
    prev_state = int(init_state)

    for k in range(N):
        Tpk = T_pred[k]
        Tsp = T_sp_series[k]

        st = heater_state_hysteresis(Tpk, Tsp, prev_state)
        state[k] = st
        prev_state = st

        Qin[k] = predict_Pon_W(T_out[k], max(Tsp - Tpk, 0.0), heater_params) if st == 1 else 0.0

        R_eff = R_occ if occ_flag[k] else R_unocc
        Ql[k] = (Tpk - T_out[k]) / R_eff

        T_pred[k + 1] = Tpk + (DT / C) * (Qin[k] - Ql[k])

    return T_pred, Qin, Ql, state

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION LOSSES
# ══════════════════════════════════════════════════════════════════════════════
def loss_explicit(log_R, data, T_measured, heater_params, init_state):
    params = {"C_pool": C_FIXED, "R_pool": np.exp(log_R[0])}
    sse = 0.0
    for start in range(0, len(data) - BLOCK, BLOCK):
        end = start + BLOCK
        T_sim, *_ = simulate_explicit(
            T0=T_measured[start],
            T_sp_series=data["T_sp"].values[start:end],
            T_air=data["T_air_C"].values[start:end],
            RH_pct=data["RH_pct"].values[start:end],
            T_out=data["T_out"].values[start:end],
            params=params,
            heater_params=heater_params,
            occ_flag=data["is_occ"].values[start:end].astype(bool),
            init_state=init_state if start == 0 else 0
        )
        sse += np.sum((T_sim[1:] - T_measured[start + 1:end + 1]) ** 2)
    return sse

def loss_singleR(log_R, data, T_measured, heater_params, init_state):
    params = {"C_pool": C_FIXED, "R_pool": np.exp(log_R[0])}
    sse = 0.0
    for start in range(0, len(data) - BLOCK, BLOCK):
        end = start + BLOCK
        T_sim, *_ = simulate_lumped_singleR(
            T0=T_measured[start],
            T_sp_series=data["T_sp"].values[start:end],
            T_out=data["T_out"].values[start:end],
            params=params,
            heater_params=heater_params,
            init_state=init_state if start == 0 else 0
        )
        sse += np.sum((T_sim[1:] - T_measured[start + 1:end + 1]) ** 2)
    return sse

def loss_twoR(log_Rs, data, T_measured, heater_params, init_state):
    params = {
        "C_pool": C_FIXED,
        "R_unocc": np.exp(log_Rs[0]),
        "R_occ": np.exp(log_Rs[1])
    }
    sse = 0.0
    for start in range(0, len(data) - BLOCK, BLOCK):
        end = start + BLOCK
        T_sim, *_ = simulate_lumped_twoR(
            T0=T_measured[start],
            T_sp_series=data["T_sp"].values[start:end],
            T_out=data["T_out"].values[start:end],
            occ_flag=data["is_occ"].values[start:end].astype(bool),
            params=params,
            heater_params=heater_params,
            init_state=init_state if start == 0 else 0
        )
        sse += np.sum((T_sim[1:] - T_measured[start + 1:end + 1]) ** 2)
    return sse

# ══════════════════════════════════════════════════════════════════════════════
# BLOCK RUNNERS
# ══════════════════════════════════════════════════════════════════════════════
def run_blocks_explicit(dataset, T_arr, params, heater_params):
    n_blocks = (len(dataset) - 1) // BLOCK
    T_sv, T_vm, Qin_all, Qe_all, Ql_all, st_all, stobs_all = [], [], [], [], [], [], []
    for i in range(n_blocks):
        s = i * BLOCK
        e = s + BLOCK
        T_sim, Qin, Qe, Ql, st = simulate_explicit(
            T_arr[s],
            dataset["T_sp"].values[s:e],
            dataset["T_air_C"].values[s:e],
            dataset["RH_pct"].values[s:e],
            dataset["T_out"].values[s:e],
            params,
            heater_params,
            dataset["is_occ"].values[s:e].astype(bool),
            int(dataset["heater_on_obs"].iloc[s])
        )
        T_sv.append(T_sim[1:])
        T_vm.append(T_arr[s + 1:e + 1])
        Qin_all.append(Qin)
        Qe_all.append(Qe)
        Ql_all.append(Ql)
        st_all.append(st)
        stobs_all.append(dataset["heater_on_obs"].values[s:e])
    return map(np.concatenate, [T_sv, T_vm, Qin_all, Qe_all, Ql_all, st_all, stobs_all])

def run_blocks_singleR(dataset, T_arr, params, heater_params):
    n_blocks = (len(dataset) - 1) // BLOCK
    T_sv, T_vm, Qin_all, Ql_all, st_all, stobs_all = [], [], [], [], [], []
    for i in range(n_blocks):
        s = i * BLOCK
        e = s + BLOCK
        T_sim, Qin, Ql, st = simulate_lumped_singleR(
            T_arr[s],
            dataset["T_sp"].values[s:e],
            dataset["T_out"].values[s:e],
            params,
            heater_params,
            int(dataset["heater_on_obs"].iloc[s])
        )
        T_sv.append(T_sim[1:])
        T_vm.append(T_arr[s + 1:e + 1])
        Qin_all.append(Qin)
        Ql_all.append(Ql)
        st_all.append(st)
        stobs_all.append(dataset["heater_on_obs"].values[s:e])
    return map(np.concatenate, [T_sv, T_vm, Qin_all, Ql_all, st_all, stobs_all])

def run_blocks_twoR(dataset, T_arr, params, heater_params):
    n_blocks = (len(dataset) - 1) // BLOCK
    T_sv, T_vm, Qin_all, Ql_all, st_all, stobs_all = [], [], [], [], [], []
    for i in range(n_blocks):
        s = i * BLOCK
        e = s + BLOCK
        T_sim, Qin, Ql, st = simulate_lumped_twoR(
            T_arr[s],
            dataset["T_sp"].values[s:e],
            dataset["T_out"].values[s:e],
            dataset["is_occ"].values[s:e].astype(bool),
            params,
            heater_params,
            int(dataset["heater_on_obs"].iloc[s])
        )
        T_sv.append(T_sim[1:])
        T_vm.append(T_arr[s + 1:e + 1])
        Qin_all.append(Qin)
        Ql_all.append(Ql)
        st_all.append(st)
        stobs_all.append(dataset["heater_on_obs"].values[s:e])
    return map(np.concatenate, [T_sv, T_vm, Qin_all, Ql_all, st_all, stobs_all])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    data = load_data()
    T_measured = data["T_pool"].values

    split_idx = (int(TRAIN_RATIO * len(data)) // BLOCK) * BLOCK
    data_cal = data.iloc[:split_idx].copy()
    data_val = data.iloc[split_idx:].copy()
    T_cal = T_measured[:split_idx]
    T_val = T_measured[split_idx:]

    print("=" * 72)
    print("LAYER 1 + 1R1C: EXPLICIT EVAPORATION VS LUMPED-R (WATER-ONLY C)")
    print("=" * 72)
    print(f"Dataset rows: {len(data)}")
    print(f"Calibration rows: {len(data_cal)}")
    print(f"Validation rows:  {len(data_val)}")
    print(f"C_pool (water only) = {C_FIXED/1e6:.3f} MJ/K")

    # Fit heater model on calibration only
    heater_params = fit_layer1_heater_model(data_cal)
    print("\nLayer 1 heater model:")
    print(f"P_on[kW] = {heater_params['pon_intercept']:.3f} "
          f"+ ({heater_params['pon_coef'][0]:.3f})*T_out "
          f"+ ({heater_params['pon_coef'][1]:.3f})*temp_error_pos")
    print(f"Clipping range = [{heater_params['pon_min_kW']:.3f}, {heater_params['pon_max_kW']:.3f}] kW")

    init_state = int(data_cal["heater_on_obs"].iloc[0])

    # Fit explicit baseline
    res_exp = minimize(
        loss_explicit,
        [np.log(3.5e-3)],
        args=(data_cal, T_cal, heater_params, init_state),
        method="L-BFGS-B",
        bounds=[(np.log(1e-6), np.log(0.1))]
    )
    params_exp = {"C_pool": C_FIXED, "R_pool": float(np.exp(res_exp.x[0]))}

    # Fit single-R lumped model
    res_single = minimize(
        loss_singleR,
        [np.log(3.5e-3)],
        args=(data_cal, T_cal, heater_params, init_state),
        method="L-BFGS-B",
        bounds=[(np.log(1e-6), np.log(0.1))]
    )
    params_single = {"C_pool": C_FIXED, "R_pool": float(np.exp(res_single.x[0]))}

    # Fit two-R lumped model
    res_two = minimize(
        loss_twoR,
        [np.log(3.5e-3), np.log(3.5e-3)],
        args=(data_cal, T_cal, heater_params, init_state),
        method="L-BFGS-B",
        bounds=[(np.log(1e-6), np.log(0.1)), (np.log(1e-6), np.log(0.1))]
    )
    params_two = {
        "C_pool": C_FIXED,
        "R_unocc": float(np.exp(res_two.x[0])),
        "R_occ": float(np.exp(res_two.x[1]))
    }

    print("\nFitted parameters:")
    print(f"Explicit baseline R        = {params_exp['R_pool']:.6e} K/W")
    print(f"Lumped single-R            = {params_single['R_pool']:.6e} K/W")
    print(f"Lumped two-R unoccupied    = {params_two['R_unocc']:.6e} K/W")
    print(f"Lumped two-R occupied      = {params_two['R_occ']:.6e} K/W")

    # Run models
    Tsim_cal_e, Tcal_m_e, Qin_cal_e, Qe_cal_e, Ql_cal_e, st_cal_e, stobs_cal_e = run_blocks_explicit(data_cal, T_cal, params_exp, heater_params)
    Tsim_val_e, Tval_m_e, Qin_val_e, Qe_val_e, Ql_val_e, st_val_e, stobs_val_e = run_blocks_explicit(data_val, T_val, params_exp, heater_params)

    Tsim_cal_s, Tcal_m_s, Qin_cal_s, Ql_cal_s, st_cal_s, stobs_cal_s = run_blocks_singleR(data_cal, T_cal, params_single, heater_params)
    Tsim_val_s, Tval_m_s, Qin_val_s, Ql_val_s, st_val_s, stobs_val_s = run_blocks_singleR(data_val, T_val, params_single, heater_params)

    Tsim_cal_t, Tcal_m_t, Qin_cal_t, Ql_cal_t, st_cal_t, stobs_cal_t = run_blocks_twoR(data_cal, T_cal, params_two, heater_params)
    Tsim_val_t, Tval_m_t, Qin_val_t, Ql_val_t, st_val_t, stobs_val_t = run_blocks_twoR(data_val, T_val, params_two, heater_params)

    # 24h forecast on last validation block
    n_val_blocks = (len(data_val) - 1) // BLOCK
    fc_start = (n_val_blocks - 1) * BLOCK
    T_meas_fc = T_val[fc_start:fc_start + BLOCK + 1]

    Tsim_fc_e, Qin_fc_e, *_ = simulate_explicit(
        T0=T_val[fc_start],
        T_sp_series=data_val["T_sp"].values[fc_start:fc_start + BLOCK],
        T_air=data_val["T_air_C"].values[fc_start:fc_start + BLOCK],
        RH_pct=data_val["RH_pct"].values[fc_start:fc_start + BLOCK],
        T_out=data_val["T_out"].values[fc_start:fc_start + BLOCK],
        params=params_exp,
        heater_params=heater_params,
        occ_flag=data_val["is_occ"].values[fc_start:fc_start + BLOCK].astype(bool),
        init_state=int(data_val["heater_on_obs"].iloc[fc_start])
    )

    Tsim_fc_s, Qin_fc_s, *_ = simulate_lumped_singleR(
        T0=T_val[fc_start],
        T_sp_series=data_val["T_sp"].values[fc_start:fc_start + BLOCK],
        T_out=data_val["T_out"].values[fc_start:fc_start + BLOCK],
        params=params_single,
        heater_params=heater_params,
        init_state=int(data_val["heater_on_obs"].iloc[fc_start])
    )

    Tsim_fc_t, Qin_fc_t, *_ = simulate_lumped_twoR(
        T0=T_val[fc_start],
        T_sp_series=data_val["T_sp"].values[fc_start:fc_start + BLOCK],
        T_out=data_val["T_out"].values[fc_start:fc_start + BLOCK],
        occ_flag=data_val["is_occ"].values[fc_start:fc_start + BLOCK].astype(bool),
        params=params_two,
        heater_params=heater_params,
        init_state=int(data_val["heater_on_obs"].iloc[fc_start])
    )

    # Metrics
    print("\n" + "─" * 72)
    print("POOL TEMPERATURE METRICS")
    print("─" * 72)

    m_cal_e = compute_metrics(Tcal_m_e, Tsim_cal_e)
    m_val_e = compute_metrics(Tval_m_e, Tsim_val_e)
    m_fc_e  = compute_metrics(T_meas_fc[1:], Tsim_fc_e[1:])

    m_cal_s = compute_metrics(Tcal_m_s, Tsim_cal_s)
    m_val_s = compute_metrics(Tval_m_s, Tsim_val_s)
    m_fc_s  = compute_metrics(T_meas_fc[1:], Tsim_fc_s[1:])

    m_cal_t = compute_metrics(Tcal_m_t, Tsim_cal_t)
    m_val_t = compute_metrics(Tval_m_t, Tsim_val_t)
    m_fc_t  = compute_metrics(T_meas_fc[1:], Tsim_fc_t[1:])

    print("\n1) Explicit evaporation + single R")
    print(f"Calibration: RMSE={m_cal_e['RMSE']:.3f} °C, MAE={m_cal_e['MAE']:.3f} °C, R²={m_cal_e['R2']:.3f}")
    print(f"Validation : RMSE={m_val_e['RMSE']:.3f} °C, MAE={m_val_e['MAE']:.3f} °C, R²={m_val_e['R2']:.3f}")
    print(f"24h forecast: RMSE={m_fc_e['RMSE']:.3f} °C, MAE={m_fc_e['MAE']:.3f} °C, R²={m_fc_e['R2']:.3f}")

    print("\n2) Lumped losses only + single R")
    print(f"Calibration: RMSE={m_cal_s['RMSE']:.3f} °C, MAE={m_cal_s['MAE']:.3f} °C, R²={m_cal_s['R2']:.3f}")
    print(f"Validation : RMSE={m_val_s['RMSE']:.3f} °C, MAE={m_val_s['MAE']:.3f} °C, R²={m_val_s['R2']:.3f}")
    print(f"24h forecast: RMSE={m_fc_s['RMSE']:.3f} °C, MAE={m_fc_s['MAE']:.3f} °C, R²={m_fc_s['R2']:.3f}")

    print("\n3) Lumped losses only + separate R_unocc / R_occ")
    print(f"Calibration: RMSE={m_cal_t['RMSE']:.3f} °C, MAE={m_cal_t['MAE']:.3f} °C, R²={m_cal_t['R2']:.3f}")
    print(f"Validation : RMSE={m_val_t['RMSE']:.3f} °C, MAE={m_val_t['MAE']:.3f} °C, R²={m_val_t['R2']:.3f}")
    print(f"24h forecast: RMSE={m_fc_t['RMSE']:.3f} °C, MAE={m_fc_t['MAE']:.3f} °C, R²={m_fc_t['R2']:.3f}")

    print("\n" + "─" * 72)
    print("STATE METRICS ON VALIDATION")
    print("─" * 72)
    for label, stobs, stpred in [
        ("Explicit", stobs_val_e, st_val_e),
        ("Single-R", stobs_val_s, st_val_s),
        ("Two-R", stobs_val_t, st_val_t),
    ]:
        acc = accuracy_score(stobs, stpred)
        prec = precision_score(stobs, stpred, zero_division=0)
        rec = recall_score(stobs, stpred, zero_division=0)
        f1 = f1_score(stobs, stpred, zero_division=0)
        print(f"{label:<10} Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    print("\nValidation confusion matrix for Single-R model:")
    print(confusion_matrix(stobs_val_s, st_val_s))