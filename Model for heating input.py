import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =========================
# 1. LOAD DATA
# =========================
file_path = r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\pool_unified_10min_with_station_temp.csv"
df = pd.read_csv(file_path)

# =========================
# 2. CLEANING
# =========================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

required_cols = [
    'timestamp',
    'power_W',
    'flow_lph',
    'diff_pressure_kPa',
    'diff_temperature_K',
    'temp_t1_C',
    'temp_t2_C',
    'Outside air RH',
    'outdoor air temperature station'
]

df = df[required_cols].copy()
df = df.dropna(subset=required_cols).reset_index(drop=True)

# =========================
# 3. SETTINGS
# =========================
flow_threshold = 100  # lph
outdoor_col = 'outdoor air temperature station'
target_temp = 'temp_t2_C'
target_power = 'power_W'

# =========================
# 4. DEFINE HEATING MODE
# =========================
df['heating_mode'] = df['flow_lph'] > flow_threshold

print("\n=== Heating Mode Counts ===")
print(df['heating_mode'].value_counts())

df_heat = df[df['heating_mode']].copy().reset_index(drop=True)

print("\n=== Heating Mode Dataset Info ===")
print("Rows in heating mode:", len(df_heat))
print("Start:", df_heat['timestamp'].min())
print("End:  ", df_heat['timestamp'].max())

# =========================
# 5. EXPLORATORY PLOTS
# =========================

# 5A. Supply and return temperature plot
plt.figure(figsize=(14, 5))
plt.plot(df['timestamp'], df['temp_t2_C'], label='Supply Water Temperature')
plt.plot(df['timestamp'], df['temp_t1_C'], label='Return Water Temperature')
plt.title('Supply and Return Water Temperature Profile')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5B. Power and mass flow on same plot
fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(df['timestamp'], df['power_W'], color='orange', label='Power (W)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Power (W)', color='orange')
ax1.tick_params(axis='y', labelcolor='orange')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df['timestamp'], df['flow_lph'], color='blue', linestyle='--', label='Mass Flow (lph)')
ax2.set_ylabel('Mass Flow (lph)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='upper right')

plt.title('Power and Mass Flow over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5C. Correlation plot: outdoor station temp vs supply water temp
df_corr = df_heat[[outdoor_col, target_temp]].dropna().copy()

corr_value = df_corr[outdoor_col].corr(df_corr[target_temp])

print("\n=== Correlation: Outdoor Station Temp vs Supply Water Temp ===")
print(f"Pearson correlation: {corr_value:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(df_corr[outdoor_col], df_corr[target_temp], alpha=0.3)
plt.xlabel('Outdoor Air Temperature Station (°C)')
plt.ylabel('Supply Water Temperature (°C)')
plt.title('Outdoor Station Temperature vs Supply Water Temperature (Heating Mode Only)')
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 6. LINEAR REGRESSION MODEL
#    Predict supply water temp
#    from outdoor station temp only
# =========================
model_df = df_heat[[outdoor_col, target_temp, 'timestamp', target_power]].dropna().copy()

# chronological split
split_idx = int(len(model_df) * 0.8)
train = model_df.iloc[:split_idx].copy()
test = model_df.iloc[split_idx:].copy()

X_train = train[[outdoor_col]]
X_test = test[[outdoor_col]]

y_train = train[target_temp]
y_test = test[target_temp]

model_temp = LinearRegression()
model_temp.fit(X_train, y_train)

# predict supply temp for train and test
train['temp_t2_pred'] = model_temp.predict(X_train)
test['temp_t2_pred'] = model_temp.predict(X_test)

rmse_temp = np.sqrt(mean_squared_error(y_test, test['temp_t2_pred']))
mae_temp = mean_absolute_error(y_test, test['temp_t2_pred'])
r2_temp = r2_score(y_test, test['temp_t2_pred'])

print("\n=== Linear Regression Model: Supply Water Temperature ===")
print(f"Input variable: {outdoor_col}")
print(f"RMSE: {rmse_temp:.4f}")
print(f"MAE:  {mae_temp:.4f}")
print(f"R2:   {r2_temp:.4f}")
print(f"Model equation: temp_t2_C = {model_temp.coef_[0]:.4f} * ({outdoor_col}) + {model_temp.intercept_:.4f}")

# =========================
# 7. MODEL PLOTS
# =========================

# 7A. Actual vs predicted over time
plt.figure(figsize=(14, 5))
plt.plot(test['timestamp'], y_test.values, label='Actual Supply Water Temp')
plt.plot(test['timestamp'], test['temp_t2_pred'], label='Predicted Supply Water Temp')
plt.title('Actual vs Predicted Supply Water Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7B. Scatter: actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test['temp_t2_pred'], alpha=0.3)
plt.xlabel('Actual Supply Water Temp (°C)')
plt.ylabel('Predicted Supply Water Temp (°C)')
plt.title('Actual vs Predicted Supply Water Temperature')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7C. Scatter with 45-degree line
min_val = min(y_test.min(), test['temp_t2_pred'].min())
max_val = max(y_test.max(), test['temp_t2_pred'].max())

plt.figure(figsize=(6, 6))
plt.scatter(y_test, test['temp_t2_pred'], alpha=0.3)
plt.plot([min_val, max_val], [min_val, max_val], '--')
plt.xlabel('Actual Supply Water Temp (°C)')
plt.ylabel('Predicted Supply Water Temp (°C)')
plt.title('Actual vs Predicted Supply Water Temperature with Reference Line')
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 8. SAVE SUPPLY TEMP MODEL RESULTS
# =========================
results_temp = test[['timestamp', outdoor_col, 'temp_t2_C', 'temp_t2_pred']].copy()

results_temp.to_csv(
    r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\supply_temp_model_station_temp_only_results.csv",
    index=False
)

print("\nSaved:")
print(r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\supply_temp_model_station_temp_only_results.csv")

# =========================
# 9. POWER MODEL
#    USING PREDICTED SUPPLY TEMP
# =========================

# Time features
train['hour'] = train['timestamp'].dt.hour
train['dayofweek'] = train['timestamp'].dt.dayofweek

test['hour'] = test['timestamp'].dt.hour
test['dayofweek'] = test['timestamp'].dt.dayofweek

features_power = [
    'temp_t2_pred',
    'outdoor air temperature station',
    'hour',
    'dayofweek'
]

X_train_power = train[features_power]
X_test_power = test[features_power]

y_train_power = train[target_power]
y_test_power = test[target_power]

model_power = LinearRegression()
model_power.fit(X_train_power, y_train_power)

test['power_pred'] = model_power.predict(X_test_power)
test['power_pred'] = np.maximum(test['power_pred'], 0)

rmse_power = np.sqrt(mean_squared_error(y_test_power, test['power_pred']))
mae_power = mean_absolute_error(y_test_power, test['power_pred'])
r2_power = r2_score(y_test_power, test['power_pred'])

print("\n=== Power Prediction Model (Heating Mode Only, Using Predicted Supply Temp) ===")
print("Features used:", features_power)
print(f"RMSE: {rmse_power:.4f} W")
print(f"MAE:  {mae_power:.4f} W")
print(f"R2:   {r2_power:.4f}")
print("Intercept:", model_power.intercept_)
print("Coefficients:")
for f, c in zip(features_power, model_power.coef_):
    print(f"  {f}: {c:.6f}")

# =========================
# 10. POWER MODEL PLOTS
# =========================

# 10A. Actual vs predicted power over time
plt.figure(figsize=(14, 5))
plt.plot(test['timestamp'], y_test_power.values, label='Actual Power')
plt.plot(test['timestamp'], test['power_pred'], label='Predicted Power')
plt.title('Actual vs Predicted Heating Power (Using Predicted Supply Temp)')
plt.xlabel('Time')
plt.ylabel('Power (W)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10B. Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test_power, test['power_pred'], alpha=0.3)
plt.xlabel('Actual Power (W)')
plt.ylabel('Predicted Power (W)')
plt.title('Actual vs Predicted Heating Power')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10C. Scatter plot with 45-degree line
min_power = min(y_test_power.min(), test['power_pred'].min())
max_power = max(y_test_power.max(), test['power_pred'].max())

plt.figure(figsize=(6, 6))
plt.scatter(y_test_power, test['power_pred'], alpha=0.3)
plt.plot([min_power, max_power], [min_power, max_power], '--')
plt.xlabel('Actual Power (W)')
plt.ylabel('Predicted Power (W)')
plt.title('Actual vs Predicted Heating Power with Reference Line')
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 11. SAVE POWER MODEL RESULTS
# =========================
results_power = test[[
    'timestamp',
    outdoor_col,
    'temp_t2_C',
    'temp_t2_pred',
    'power_W',
    'power_pred',
    'hour',
    'dayofweek'
]].copy()

results_power.to_csv(
    r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\power_model_using_predicted_supply_temp_results.csv",
    index=False
)

print("\nSaved:")
print(r"M:\PhD\03 Experiments\Dalgård\Energy meter data\Pool\power_model_using_predicted_supply_temp_results.csv")

# =========================
# AVERAGE 24-HOUR PROFILE FOR EACH DAY OF WEEK
# for:
# - power_W
# - temp_t2_C
# =========================

df_profile = df.copy()

# Time features
df_profile['hour'] = df_profile['timestamp'].dt.hour
df_profile['dayofweek'] = df_profile['timestamp'].dt.dayofweek

# Map weekday names
day_map = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}
df_profile['day_name'] = df_profile['dayofweek'].map(day_map)

# Average profile by hour for each weekday
avg_power_profile = df_profile.groupby(['day_name', 'hour'])['power_W'].mean().unstack(0)
avg_temp_profile = df_profile.groupby(['day_name', 'hour'])['temp_t2_C'].mean().unstack(0)

# Reorder columns
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
avg_power_profile = avg_power_profile[ordered_days]
avg_temp_profile = avg_temp_profile[ordered_days]

# -------------------------
# PLOT: AVERAGE POWER PROFILE
# -------------------------
plt.figure(figsize=(12, 6))
for day in ordered_days:
    plt.plot(avg_power_profile.index, avg_power_profile[day], label=day)

plt.xlabel('Hour of Day')
plt.ylabel('Average Power (W)')
plt.title('Average 24-Hour Power Profile for Each Day of Week')
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# PLOT: AVERAGE SUPPLY TEMP PROFILE
# -------------------------
plt.figure(figsize=(12, 6))
for day in ordered_days:
    plt.plot(avg_temp_profile.index, avg_temp_profile[day], label=day)

plt.xlabel('Hour of Day')
plt.ylabel('Average Supply Water Temperature (°C)')
plt.title('Average 24-Hour Supply Water Temperature Profile for Each Day of Week')
plt.xticks(range(24))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()