import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    USE_XGB = True
    print("XGBoost found — using XGBRegressor / XGBClassifier.")
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    USE_XGB = False
    print("XGBoost not found — falling back to sklearn GradientBoosting.")

# All plots are saved here with fixed names — each run overwrites the previous files.
PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# 1. LOAD & RESAMPLE
# =========================
file_path = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\Pool_heater_with_weather_10min_.csv"
df = pd.read_csv(file_path, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Resample to hourly averages — reduces noise and matches MPC control horizon
df = df.set_index('timestamp').resample('h').mean().reset_index()

n_days = (df['timestamp'].max() - df['timestamp'].min()).days
print(f"Dataset: {len(df)} hourly rows  |  {df['timestamp'].min()} → {df['timestamp'].max()}  ({n_days} days)")

if n_days < 30:
    print(f"\n  WARNING: Only {n_days} days of data available.")
    print("  The 7-day lag and weekly profile features will be imputed from shorter lags.")
    print("  For a robust day-ahead model, aim for at least 3-6 months of historical data.\n")

# =========================
# 1b. DR EXPERIMENT SETPOINT LOG
#     The pool water temperature setpoint was manually changed during demand
#     response (DR) experiments. The sensor data does not contain this information,
#     so without it the model sees unexplained power drops/spikes and learns the
#     wrong pattern.
#
#     Crucially, setpoint is a PLANNED, CONTROLLABLE input: the MPC itself decides
#     when to run DR events, so the setpoint is always known day-ahead and is a
#     legitimate feature for day-ahead forecasting.
#
#     The log below was provided manually. One gap exists: the return from 31.5 → 33.5
#     between 19/02 and 23/02 is not explicitly logged. Event on 23/02 is described as
#     "from 33.5 to 31.5", implying the setpoint had recovered to 33.5 by then.
#     We infer the recovery occurred over the weekend (Saturday 21/02 08:00) — flagged
#     with a comment below. Adjust this timestamp if you have a more precise record.
# =========================
dr_events = pd.DataFrame([
    # (timestamp,              setpoint °C)   # note
    ('2026-02-18 15:00', 33.5),               # heater turned ON → 33.5
    ('2026-02-19 12:10', 31.5),               # reduced → 31.5
    ('2026-02-21 08:00', 33.5),               # INFERRED: recovery to normal over weekend
    ('2026-02-23 10:49', 31.5),               # explicit: 33.5 → 31.5
    ('2026-02-26 14:32', 33.5),
    ('2026-02-27 08:10', 31.5),
    ('2026-02-27 14:31', 33.5),
    ('2026-03-02 08:02', 31.5),
    ('2026-03-02 14:31', 33.5),
    ('2026-03-03 08:07', 31.5),
    ('2026-03-03 12:15', 33.5),
    ('2026-03-03 14:30', 31.5),
    ('2026-03-04 08:31', 33.5),
    ('2026-03-05 08:19', 31.5),
    ('2026-03-05 14:55', 33.5),
], columns=['timestamp', 'setpoint'])
dr_events['timestamp'] = pd.to_datetime(dr_events['timestamp']).dt.floor('h')
dr_events = dr_events.sort_values('timestamp').reset_index(drop=True)

# Build a continuous hourly setpoint series by forward-filling from the event log.
# Any period before the first logged event defaults to 33.5 (normal operation).
setpoint_range  = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), freq='h')
setpoint_series = pd.DataFrame({'timestamp': setpoint_range})
setpoint_series = setpoint_series.merge(
    dr_events.rename(columns={'setpoint': '_sp_raw'}),
    on='timestamp', how='left'
)
setpoint_series['setpoint'] = setpoint_series['_sp_raw'].ffill().fillna(33.5)
setpoint_series = setpoint_series[['timestamp', 'setpoint']]

# Merge into main dataframe
df = df.merge(setpoint_series, on='timestamp', how='left')
df['setpoint'] = df['setpoint'].fillna(33.5)

# Binary flag: 1 = low setpoint (31.5 °C, DR reduction), 0 = normal (33.5 °C)
df['setpoint_low'] = (df['setpoint'] < 33.0).astype(int)

# Sanity-check: print fraction of time in each mode
n_low  = df['setpoint_low'].sum()
n_high = (df['setpoint_low'] == 0).sum()
print(f"Setpoint low (31.5 °C): {n_low} hours  |  normal (33.5 °C): {n_high} hours")

# =========================
# 1c. SESSION SCHEDULE
#     Each row in the log is one day with zero or more swim sessions given as
#     "HH:MM-HH:MM" ranges. We expand these into four hourly features:
#
#       session_active       — a session is happening during this hour
#       session_upcoming_1h  — a session starts within the next 1 hour
#                              (immediate pre-heat ramp-up)
#       session_upcoming_2h  — a session starts within the next 2 hours
#                              (early pre-heat)
#       hours_since_last_session — time since the last session ended
#                              (pool cooling context); capped at 48h
#       sessions_today       — total sessions planned for this day
#
#     For day-ahead MPC use: the facility is assumed to confirm the next
#     day's session schedule by the evening before, so all five features
#     are valid day-ahead inputs.
# =========================
SESSION_LOG_PATH = r'C:\Users\mtasi\Downloads\holding_tank_sessions_full_daily_log.csv'
session_log = pd.read_csv(SESSION_LOG_PATH)
session_log['date'] = pd.to_datetime(session_log['date'])

# Parse every "HH:MM-HH:MM" string into (start_datetime, end_datetime) pairs
session_intervals = []
for _, row in session_log.iterrows():
    if row['n_sessions'] == 0:
        continue
    for interval in str(row['sessions']).split('; '):
        interval = interval.strip()
        if '-' not in interval:
            continue
        start_str, end_str = interval.split('-')
        start_dt = pd.to_datetime(str(row['date'].date()) + ' ' + start_str.strip())
        end_dt   = pd.to_datetime(str(row['date'].date()) + ' ' + end_str.strip())
        session_intervals.append((start_dt, end_dt))

session_intervals.sort(key=lambda x: x[0])

def _session_features(ts):
    """Return session features for a single hourly timestamp."""
    h_start = ts
    h_end   = ts + pd.Timedelta(hours=1)

    # Is a session active during any part of this hour?
    active = int(any(s < h_end and e > h_start for s, e in session_intervals))

    # Does a session start within the next 1 / 2 hours?
    up_1h = int(any(h_end   <= s < h_end + pd.Timedelta(hours=1) for s, e in session_intervals))
    up_2h = int(any(h_end   <= s < h_end + pd.Timedelta(hours=2) for s, e in session_intervals))

    # Hours elapsed since the most recent session ended (capped at 48)
    past_ends = [e for s, e in session_intervals if e <= h_start]
    hours_since = min((h_start - max(past_ends)).total_seconds() / 3600, 48) if past_ends else 48.0

    # Total sessions on the same calendar day
    today_count = sum(1 for s, _ in session_intervals if s.date() == ts.date())

    return active, up_1h, up_2h, hours_since, today_count

session_cols = ['session_active', 'session_upcoming_1h', 'session_upcoming_2h',
                'hours_since_last_session', 'sessions_today']
df[session_cols] = df['timestamp'].apply(
    lambda ts: pd.Series(_session_features(ts), index=session_cols)
)

print(f"Session features added.")
print(f"  Hours with active session : {df['session_active'].sum()}")
print(f"  Hours with session in 1h  : {df['session_upcoming_1h'].sum()}")
print(f"  Hours with session in 2h  : {df['session_upcoming_2h'].sum()}")

# =========================
# 2. FEATURE ENGINEERING
# =========================
outdoor_col = 'Outside air temp from station'

# --- Calendar / time features ---
df['hour']       = df['timestamp'].dt.hour
df['dayofweek']  = df['timestamp'].dt.dayofweek
df['month']      = df['timestamp'].dt.month
df['is_workday'] = (df['dayofweek'] < 5).astype(int)

# Cyclic encoding (avoids artificial discontinuity at midnight / year-end)
df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- Power lag features ---
# Only 24h and 48h are required; 168h is optional (needs 7+ days of clean data)
df['power_lag_24h']  = df['power_W'].shift(24)
df['power_lag_48h']  = df['power_W'].shift(48)
df['power_lag_168h'] = df['power_W'].shift(168)   # will be NaN when dataset < 8 days

# --- Temperature lag features (all valid day-ahead — based on yesterday's readings) ---
# These replace Stage 1 (supply temp model): instead of predicting supply temp from
# outdoor temp, we use yesterday's actual supply/return/pool readings directly.
# This captures the DH heat curve effect AND the DR setpoint drops without any
# error compounding from a two-stage pipeline.
df['temp_t2_lag_24h']   = df['temp_t2_C'].shift(24)           # yesterday's DH supply temp
df['temp_t1_lag_24h']   = df['temp_t1_C'].shift(24)           # yesterday's DH return temp
df['diff_temp_lag_24h'] = df['diff_temperature_K'].shift(24)  # yesterday's ΔT (supply−return)
df['pool_temp_lag_24h'] = df['pool temperature'].shift(24)    # yesterday's pool water temp
df['pool_temp_lag_48h'] = df['pool temperature'].shift(48)    # 2 days ago pool water temp

# --- Historical profile: rolling 7-occurrence mean at same (hour, weekday) ---
# Requires ~7 weeks of data to be meaningful; will be NaN for short datasets.
df = df.sort_values('timestamp').reset_index(drop=True)
df['rolling_profile_7w'] = (
    df.groupby(['hour', 'dayofweek'])['power_W']
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=2).mean())
)

# --- Heating mode label ---
# Based on power_W rather than flow_lph: power survives hourly resampling correctly.
# An intermittent pump (e.g. 10 min on / 50 min off) averages to ~67 lph after
# resampling, which would fall below a 100 lph flow threshold even though heat was
# supplied. Power does not have this problem.
power_heat_threshold = 100  # W
df['heating_mode'] = (df['power_W'] > power_heat_threshold).astype(int)

print(f"Heating-on fraction (power > {power_heat_threshold} W): {df['heating_mode'].mean():.2%}")

# =========================
# 3. CLEAN — DROP NaN
#    Only require the shortest lags so that datasets as short as 3 days are usable.
#    Longer lags (168h, rolling profile) are imputed rather than dropped.
# =========================
required_always = [
    'power_W', outdoor_col, 'temp_t2_C', 'temp_t1_C',
    'power_lag_24h', 'power_lag_48h',
    'pool_temp_lag_24h',   # pool temp starts same time as dataset; lag-24h always available
]
df_clean = df.dropna(subset=required_always).reset_index(drop=True)

# Impute longer lags where unavailable (e.g. start of dataset)
df_clean['power_lag_168h']    = df_clean['power_lag_168h'].fillna(df_clean['power_lag_24h'])
df_clean['rolling_profile_7w'] = df_clean['rolling_profile_7w'].fillna(df_clean['power_lag_24h'])
df_clean['pool_temp_lag_48h'] = df_clean['pool_temp_lag_48h'].fillna(df_clean['pool_temp_lag_24h'])
df_clean['diff_temp_lag_24h'] = df_clean['diff_temp_lag_24h'].fillna(0.0)

print(f"After cleaning: {len(df_clean)} samples")
print(f"  Heating-on: {df_clean['heating_mode'].sum()}  |  Off: {(df_clean['heating_mode'] == 0).sum()}")

# =========================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
#    Fixed at 1 March so the test window contains the dense DR events on
#    2–5 March. The training set still sees the earlier DR events (23 Feb,
#    26–27 Feb), so the model learns the setpoint signal before being
#    evaluated on unseen DR periods.
# =========================
split_date = pd.Timestamp('2026-03-01')
train = df_clean[df_clean['timestamp'] < split_date].copy()
test  = df_clean[df_clean['timestamp'] >= split_date].copy()

print(f"\nTrain: {len(train)} rows  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"  Heating-on in train:    {train['heating_mode'].sum()} / {len(train)}")
print(f"  DR (setpoint low) rows: {train['setpoint_low'].sum()} / {len(train)}")
print(f"Test:  {len(test)} rows  ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")
print(f"  Heating-on in test:     {test['heating_mode'].sum()} / {len(test)}")
print(f"  DR (setpoint low) rows: {test['setpoint_low'].sum()} / {len(test)}")

if train['heating_mode'].sum() == 0:
    raise ValueError(
        "No heating-mode samples in the training set. "
        f"Try lowering power_heat_threshold (currently {power_heat_threshold} W) "
        "or verify that the dataset covers periods with active pool heating."
    )
if test['heating_mode'].sum() == 0:
    print("WARNING: No heating-mode samples in the test set — evaluation on heating-on "
          "periods will be skipped.")

# =========================
# 5. FEATURE LIST FOR STAGES 2A & 2B
#    Stage 1 (supply temperature model) has been removed. With only a few weeks
#    of single-season data, supply_temp_pred was essentially a noisy transformation
#    of outdoor_temp — information the gradient boosting model can learn directly.
#    Removing it eliminates error compounding from a two-stage pipeline.
#    Re-add supply_temp_pred here once the dataset spans multiple seasons.
# =========================
clf_reg_features = [
    # Weather
    outdoor_col,
    # Time
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dayofweek', 'is_workday',
    # Power history
    'power_lag_24h', 'power_lag_48h', 'power_lag_168h',
    'rolling_profile_7w',
    # Temperature history — replaces Stage 1; captures DH heat curve + DR drops
    # without error compounding. All are yesterday's actual readings (valid day-ahead).
    'temp_t2_lag_24h',    # DH supply temp yesterday same hour
    'temp_t1_lag_24h',    # DH return temp yesterday same hour
    'diff_temp_lag_24h',  # ΔT (supply−return) yesterday same hour
    'pool_temp_lag_24h',  # pool water temp yesterday same hour (primary demand driver)
    'pool_temp_lag_48h',  # pool water temp 2 days ago same hour
    # DR setpoint — always known day-ahead because the MPC plans its own events
    'setpoint',
    'setpoint_low',
    # Session schedule — known day-ahead from facility booking system
    'session_active',        # is a session happening this hour?
    'session_upcoming_1h',   # session starts within next 1h (immediate pre-heat)
    'session_upcoming_2h',   # session starts within next 2h (early pre-heat)
    'hours_since_last_session',  # pool cooling context since last session
    'sessions_today',        # total sessions planned today
]

# Reduce model complexity when data is small to avoid overfitting
small_dataset = len(df_clean) < 500
n_estimators  = 100 if small_dataset else 300
max_depth     = 3   if small_dataset else 5
min_samples   = 5   if small_dataset else 10

# =========================
# 7. STAGE 2A: HEATING MODE CLASSIFIER
#    Predicts whether the pool heating system will be active (day-ahead).
# =========================
X_train_cls = train[clf_reg_features]
X_test_cls  = test[clf_reg_features]
y_train_cls = train['heating_mode']
y_test_cls  = test['heating_mode']

# Handle class imbalance
neg = int((y_train_cls == 0).sum())
pos = int((y_train_cls == 1).sum())
spw = neg / pos if pos > 0 else 1.0

if USE_XGB:
    classifier = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric='logloss', verbosity=0, random_state=42
    )
else:
    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=min_samples, random_state=42
    )

classifier.fit(X_train_cls, y_train_cls)
y_pred_cls = classifier.predict(X_test_cls)
y_prob_cls = classifier.predict_proba(X_test_cls)[:, 1]

print("\n=== Stage 2A: Heating Mode Classifier ===")
print(classification_report(y_test_cls, y_pred_cls, target_names=['Off', 'Heating']))

# =========================
# 8. STAGE 2B: POWER REGRESSOR
#    Trained on heating-mode rows only — avoids biasing the model toward zero
#    during off periods. Predicts power *given* heating is active.
# =========================
train_heat = train[train['heating_mode'] == 1].copy()

if USE_XGB:
    regressor = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
else:
    regressor = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=min_samples, random_state=42
    )

regressor.fit(train_heat[clf_reg_features], train_heat['power_W'])
power_given_heating = np.maximum(regressor.predict(test[clf_reg_features]), 0)

# =========================
# 9. COMBINED PREDICTION
#    P_pred = P(heating_on) × P_given_heating_on
#    Naturally tapers to zero when the classifier is confident heating is off.
# =========================
test = test.copy()
test['heating_prob']             = y_prob_cls
test['power_given_heating_pred'] = power_given_heating
test['power_pred']               = y_prob_cls * power_given_heating

# Also generate predictions on the TRAINING set — used only for visualisation
# (in-sample), not for honest model evaluation.
train = train.copy()
train['heating_prob']   = classifier.predict_proba(train[clf_reg_features])[:, 1]
train['power_pred']     = train['heating_prob'] * np.maximum(
    regressor.predict(train[clf_reg_features]), 0
)

# -----------------------------------------------------------------------
# HARD DR OVERRIDE
# The DR log is ground truth: when setpoint_low == 1 the heater was
# explicitly commanded off by the experimenter. Whatever the model
# predicts, the actual supplied power during those periods is zero.
# We enforce this as a post-processing constraint on all predictions so
# the final output never contradicts a known operational fact.
# -----------------------------------------------------------------------
test['power_pred_model']  = test['power_pred'].copy()     # keep raw model output for reference
test['power_pred']        = np.where(test['setpoint_low'] == 1,  0.0, test['power_pred'])

train['power_pred_model'] = train['power_pred'].copy()
train['power_pred']       = np.where(train['setpoint_low'] == 1, 0.0, train['power_pred'])

# Combine for full-dataset view
full = pd.concat([train, test], ignore_index=True).sort_values('timestamp')

# Metrics — full test set (after override)
rmse_all = np.sqrt(mean_squared_error(test['power_W'], test['power_pred']))
mae_all  = mean_absolute_error(test['power_W'], test['power_pred'])
r2_all   = r2_score(test['power_W'], test['power_pred'])

print("\n=== Combined Day-Ahead Power Model (with DR override) ===")
print(f"Full test set:      RMSE={rmse_all:.1f} W  |  MAE={mae_all:.1f} W  |  R²={r2_all:.4f}")

# Metrics — normal operation only in test (setpoint_low == 0)
test_normal = test[test['setpoint_low'] == 0]
if len(test_normal) > 0:
    rmse_normal = np.sqrt(mean_squared_error(test_normal['power_W'], test_normal['power_pred']))
    mae_normal  = mean_absolute_error(test_normal['power_W'], test_normal['power_pred'])
    r2_normal   = r2_score(test_normal['power_W'], test_normal['power_pred'])
    print(f"Normal operation:   RMSE={rmse_normal:.1f} W  |  MAE={mae_normal:.1f} W  |  R²={r2_normal:.4f}")

# Metrics — heating-on periods only (the MPC-relevant subset)
test_on = test[test['heating_mode'] == 1]
if len(test_on) > 0:
    rmse_on = np.sqrt(mean_squared_error(test_on['power_W'], test_on['power_pred']))
    mae_on  = mean_absolute_error(test_on['power_W'], test_on['power_pred'])
    r2_on   = r2_score(test_on['power_W'], test_on['power_pred'])
    print(f"Heating-on only:    RMSE={rmse_on:.1f} W  |  MAE={mae_on:.1f} W  |  R²={r2_on:.4f}")
else:
    rmse_on = mae_on = r2_on = float('nan')
    print("Heating-on only:    (no heating-on samples in test set)")

# Metrics — DR periods (train + test, in-sample for most events)
dr_all = full[full['setpoint_low'] == 1]
if len(dr_all) > 0:
    # After override, power_pred is always 0 in DR periods, so RMSE == mean actual power
    # during those periods — this tells us how much energy the DR events saved.
    mean_actual_dr = dr_all['power_W'].mean()
    print(f"\nDR periods (31.5°C, n={len(dr_all)} hours): "
          f"mean actual power = {mean_actual_dr:.1f} W  "
          f"(override sets prediction to 0 for all these hours)")
else:
    print("DR periods: none found in dataset (check setpoint log timestamps)")

print("\n=== Feature Importance (Power Regressor) ===")
importances = pd.Series(
    regressor.feature_importances_, index=clf_reg_features
).sort_values(ascending=False)
print(importances.to_string())

# =========================
# 10. PLOTS
# =========================

# 10A. Supply, return and pool water temperature (informational)
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(full['timestamp'], full['temp_t2_C'],        label='DH Supply Temp (°C)',  color='tomato',     linewidth=1.2)
ax.plot(full['timestamp'], full['temp_t1_C'],        label='DH Return Temp (°C)',  color='steelblue',  linewidth=1.2)
ax.plot(full['timestamp'], full['pool temperature'], label='Pool Water Temp (°C)', color='seagreen',   linewidth=1.5)
ax.set_title('DH Supply / Return and Pool Water Temperature (full dataset)')
ax.set_ylabel('Temperature (°C)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot_01_temperatures.png'), dpi=150)
plt.show()

# 10B. Full-dataset power: actual vs predicted, DR periods shaded
#      Train predictions are IN-SAMPLE (shown for context only).
#      Test predictions are OUT-OF-SAMPLE (honest evaluation).
fig, ax = plt.subplots(figsize=(16, 5))

# Shade DR (setpoint_low) periods across the full timeline
dr_on = False
dr_start = None
for _, row in full[['timestamp', 'setpoint_low']].iterrows():
    if row['setpoint_low'] == 1 and not dr_on:
        dr_start = row['timestamp']
        dr_on = True
    elif row['setpoint_low'] == 0 and dr_on:
        ax.axvspan(dr_start, row['timestamp'], color='gold', alpha=0.25,
                   label='DR period (setpoint 31.5°C)' if dr_start == full[full['setpoint_low'] == 1]['timestamp'].iloc[0] else '')
        dr_on = False
if dr_on:  # close any open span at end of data
    ax.axvspan(dr_start, full['timestamp'].iloc[-1], color='gold', alpha=0.25)

# Vertical line marking train/test boundary
train_end_ts = train['timestamp'].max()
ax.axvline(train_end_ts, color='grey', linestyle=':', linewidth=1.5, label='Train / Test split')

ax.plot(full['timestamp'], full['power_W'],
        label='Actual Power', color='steelblue', linewidth=1.5)
ax.plot(full['timestamp'], full['power_pred'],
        label='Predicted Power', color='crimson', linestyle='--', linewidth=1.5)

ax.set_title(f'Heating Power — Full Dataset  |  Test: RMSE={rmse_all:.0f} W, R²={r2_all:.3f}  '
             f'(shaded = DR periods, dotted = train/test boundary)')
ax.set_xlabel('Time')
ax.set_ylabel('Thermal Power (W)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot_02_power_forecast.png'), dpi=150)
plt.show()

# 10C. Scatter: actual vs predicted (heating-on only)
if len(test_on) > 1:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_on['power_W'], test_on['power_pred'], alpha=0.4, color='steelblue')
    lims = [test_on['power_W'].min(), test_on['power_W'].max()]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual Power (W)')
    ax.set_ylabel('Predicted Power (W)')
    ax.set_title(f'Actual vs Predicted — Heating On  (R²={r2_on:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_03_scatter.png'), dpi=150)
    plt.show()

# 10D. Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
importances.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('Feature Importance — Power Regressor')
ax.set_ylabel('Importance Score')
plt.xticks(rotation=40, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'plot_04_feature_importance.png'), dpi=150)
plt.show()

# 10E. Average 24h profile: actual vs predicted, by weekday
test_cp = test.copy()
test_cp['day_name'] = test_cp['timestamp'].dt.day_name()
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

days_present = [d for d in ordered_days if d in test_cp['day_name'].values]
if days_present:
    ncols = min(4, len(days_present))
    nrows = (len(days_present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.array(axes).flatten()
    for i, day in enumerate(days_present):
        day_data = test_cp[test_cp['day_name'] == day]
        actual_profile = day_data.groupby('hour')['power_W'].mean()
        pred_profile   = day_data.groupby('hour')['power_pred'].mean()
        axes[i].plot(actual_profile.index, actual_profile.values,
                     label='Actual', color='steelblue', linewidth=2)
        axes[i].plot(pred_profile.index, pred_profile.values,
                     label='Predicted', color='crimson', linestyle='--', linewidth=2)
        axes[i].set_title(day)
        axes[i].set_xlabel('Hour')
        axes[i].set_ylabel('Avg Power (W)')
        axes[i].legend(fontsize=7)
        axes[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Average 24h Heating Profile: Actual vs Day-Ahead Predicted (by Weekday)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'plot_05_weekday_profiles.png'), dpi=150)
    plt.show()

# =========================
# 11. ROLLING-WINDOW CROSS-VALIDATION
#     Simulates realistic day-ahead forecasting:
#     train on past, predict the next chunk, slide forward.
#     Number of folds is capped to ensure each fold has enough training data.
# =========================
print("\n=== Rolling-Window Cross-Validation (Day-Ahead Simulation) ===")

# Need at least 48 training rows (2 days) per fold; scale folds to data size
n_folds   = min(5, max(2, len(df_clean) // 48 - 1))
fold_size = len(df_clean) // (n_folds + 1)
print(f"Using {n_folds} folds  (fold size = {fold_size} hours each)")

cv_results = []

for fold in range(n_folds):
    train_end  = fold_size + fold * fold_size
    test_start = train_end
    test_end   = test_start + fold_size
    if test_end > len(df_clean):
        break

    cv_tr = df_clean.iloc[:train_end].copy()
    cv_te = df_clean.iloc[test_start:test_end].copy()

    cv_tr_heat = cv_tr[cv_tr['heating_mode'] == 1]
    if len(cv_tr_heat) < 5:
        print(f"  Fold {fold + 1}: skipped (too few heating-on rows in training split)")
        continue

    # Classifier
    neg_cv = int((cv_tr['heating_mode'] == 0).sum())
    pos_cv = int((cv_tr['heating_mode'] == 1).sum())
    spw_cv = neg_cv / pos_cv if pos_cv > 0 else 1.0

    if USE_XGB:
        cv_cls = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw_cv,
            eval_metric='logloss', verbosity=0, random_state=42
        )
    else:
        cv_cls = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=min_samples, random_state=42
        )
    cv_cls.fit(cv_tr[clf_reg_features], cv_tr['heating_mode'])
    cv_prob = cv_cls.predict_proba(cv_te[clf_reg_features])[:, 1]

    # Regressor
    cv_tr_heat2 = cv_tr[cv_tr['heating_mode'] == 1]
    if USE_XGB:
        cv_reg = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
    else:
        cv_reg = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=min_samples, random_state=42
        )
    cv_reg.fit(cv_tr_heat2[clf_reg_features], cv_tr_heat2['power_W'])
    cv_power_on = np.maximum(cv_reg.predict(cv_te[clf_reg_features]), 0)
    cv_pred = cv_prob * cv_power_on

    fold_rmse = np.sqrt(mean_squared_error(cv_te['power_W'], cv_pred))
    fold_mae  = mean_absolute_error(cv_te['power_W'], cv_pred)
    fold_r2   = r2_score(cv_te['power_W'], cv_pred)

    cv_results.append({'fold': fold + 1, 'RMSE': fold_rmse, 'MAE': fold_mae, 'R2': fold_r2})
    print(f"  Fold {fold + 1}:  RMSE={fold_rmse:.1f} W  |  MAE={fold_mae:.1f} W  |  R²={fold_r2:.4f}")

if cv_results:
    cv_df = pd.DataFrame(cv_results)
    print(f"\n  Mean:  RMSE={cv_df['RMSE'].mean():.1f} W  |  MAE={cv_df['MAE'].mean():.1f} W  |  R²={cv_df['R2'].mean():.4f}")
    print(f"  Std:   RMSE={cv_df['RMSE'].std():.1f} W   |  MAE={cv_df['MAE'].std():.1f} W")
    if n_days < 30:
        print("\n  NOTE: With < 30 days of data, CV results are indicative only.")
        print("  Retrain on a full seasonal dataset for production use in the MPC.")
else:
    print("  No folds completed — dataset too small for cross-validation.")

# =========================
# 12. SAVE RESULTS
# =========================
results_out = test[[
    'timestamp', outdoor_col,
    'temp_t2_C', 'temp_t1_C',
    'power_W', 'power_pred', 'power_pred_model',
    'heating_mode', 'heating_prob',
    'setpoint', 'setpoint_low',
    'hour', 'dayofweek', 'month'
]].copy()

output_path = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\day_ahead_power_model_results.csv"
results_out.to_csv(output_path, index=False)
print(f"\nResults saved to:\n  {output_path}")