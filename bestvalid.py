
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil, os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# ──────────────────────────────────────────────
# 1. Load and prepare the data
# ──────────────────────────────────────────────
raw = pd.read_csv("data/3ármars.csv")

# Select target (wind speed 'f') and past covariates
# Wind direction 'd' is circular (0–360°), so encode as sin/cos
raw["d_sin"] = np.sin(np.radians(raw["d"]))
raw["d_cos"] = np.cos(np.radians(raw["d"]))

COVARIATE_COLS = ["t", "p", "rh", "fg", "d_sin", "d_cos"]
TARGET_COL = "f"

df = raw[["time", TARGET_COL] + COVARIATE_COLS].copy()
df.rename(columns={"time": "timestamp", TARGET_COL: "target"}, inplace=True)

# Parse timestamps, strip timezone (AutoGluon requires plain datetime64), and sort
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Raw data: {len(df)} rows")
print(f"  NaN in target (f): {df['target'].isna().sum()}")
for col in COVARIATE_COLS:
    print(f"  NaN in {col:6s}: {df[col].isna().sum()}")

# ──────────────────────────────────────────────
# 2. Re-index to a regular 10-min grid and interpolate ALL gaps
# ──────────────────────────────────────────────
full_range = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="10min",
)

df.set_index("timestamp", inplace=True)
df = df.reindex(full_range)
df.index.name = "timestamp"

# Interpolate all columns (target + covariates)
for col in ["target"] + COVARIATE_COLS:
    df[col] = df[col].interpolate(method="linear")

# Clamp: wind speed and gust cannot be negative
df["target"] = df["target"].clip(lower=0.0)
df["fg"] = df["fg"].clip(lower=0.0)

df.reset_index(inplace=True)

print(f"\nAfter interpolation: {len(df)} rows, NaN remaining:")
for col in ["target"] + COVARIATE_COLS:
    print(f"  {col:8s}: {df[col].isna().sum()}")

# ──────────────────────────────────────────────
# 3. Convert to AutoGluon TimeSeriesDataFrame
# ──────────────────────────────────────────────
df["item_id"] = "Eyrarbakki"

full_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
)

print(f"\nColumns in TimeSeriesDataFrame: {list(full_data.columns)}")

# ──────────────────────────────────────────────
# 4. Train / test split
# ──────────────────────────────────────────────
PREDICTION_LENGTH = 144   # 144 × 10 min = 24 hours

test_data = full_data.copy()
train_data = full_data.slice_by_timestep(None, -PREDICTION_LENGTH)

print(f"\nTrain rows: {len(train_data)}")
print(f"Test rows:  {len(test_data)}  (last {PREDICTION_LENGTH} steps held out)")

# ──────────────────────────────────────────────
# 5. Configure and train the predictor
# ──────────────────────────────────────────────
MODEL_PATH = "autogluon-eyrarbakki-wind"
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)

predictor = TimeSeriesPredictor(
    prediction_length=PREDICTION_LENGTH,
    path=MODEL_PATH,
    target="target",
    eval_metric="MASE",
)

# Models:
#   - Naive                       : baseline
#   - DirectTabular (RF, XGB)     : tree-based (uses lag features of target)
#   - RecursiveTabular (RF, XGB)  : tree-based recursive
#   - TemporalFusionTransformer   : deep learning, CAN use past covariates
#   - DeepAR                      : deep learning, CAN use past covariates
#   - WeightedEnsemble            : combines all of the above
#
# The covariates (t, p, rh, fg, d_sin, d_cos) are automatically detected
# as past covariates by AutoGluon since they are extra columns in the
# TimeSeriesDataFrame and are NOT listed in known_covariates_names.
predictor.fit(
    train_data,
    hyperparameters={
        "Naive": {},
        "DirectTabular": [
            {"model_name": "RF"},
            {"model_name": "XGB"},
        ],
        "TemporalFusionTransformer": {},
        "DeepAR": {},
    },
    enable_ensemble=True,
    num_val_windows=3,
    time_limit=1800,       # 20 minutes — needed for larger datasets so DL models converge
)

# ──────────────────────────────────────────────
# 6. Leaderboard (scored against held-out test data)
# ──────────────────────────────────────────────
leaderboard = predictor.leaderboard(test_data, extra_metrics=["MAE", "RMSE"])
print("\n── Model Leaderboard (test score) ──")
print(leaderboard.to_string())

# ──────────────────────────────────────────────
# 7. Generate forecast using the best model
# ──────────────────────────────────────────────
# We use the best VALIDATION model (averaged over multiple windows) rather
# than best single test window, since a single 24h window can be misleading.
best_val_model = leaderboard.sort_values("score_val", ascending=False).iloc[0]["model"]
best_test_model = leaderboard.iloc[0]["model"]
print(f"\nBest validation model: {best_val_model}  (score_val: {leaderboard.set_index('model').loc[best_val_model, 'score_val']:.4f})")
print(f"Best test model:       {best_test_model}  (score_test: {leaderboard.iloc[0]['score_test']:.4f})")
print(f"→ Using: {best_val_model} (more robust across multiple windows)")

predictions = predictor.predict(train_data, model=best_val_model)
print("\n── Forecast (first rows) ──")
print(predictions.head(10))

# ──────────────────────────────────────────────
# 8. Plot: actuals vs forecast
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

context_days = 5
context_steps = 144 * context_days
all_actuals = full_data.loc["Eyrarbakki"].tail(context_steps)

cutoff = train_data.loc["Eyrarbakki"].index[-1]
actuals_train = all_actuals[all_actuals.index <= cutoff]
actuals_test  = all_actuals[all_actuals.index > cutoff]

ax.plot(actuals_train.index, actuals_train["target"],
        label="Actual (train)", color="tab:blue", linewidth=1)
ax.plot(actuals_test.index, actuals_test["target"],
        label="Actual (held out)", color="black", linewidth=1.2)

pred = predictions.loc["Eyrarbakki"]
ax.plot(pred.index, pred["mean"],
        label=f"Forecast – {best_val_model}", color="tab:orange", linewidth=1.2)

ax.fill_between(
    pred.index,
    pred["0.1"].clip(lower=0),
    pred["0.9"],
    color="tab:orange", alpha=0.2, label="10%–90% interval",
)

ax.set_title(f"Eyrarbakki Wind Speed – Forecast vs Actual ({PREDICTION_LENGTH} steps = 24 h)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Wind Speed (m/s)")
ax.legend()
plt.tight_layout()
plt.savefig("wind_forecast_plot.png", dpi=150)
plt.show()

print("\nDone! Plot saved to wind_forecast_plot.png")