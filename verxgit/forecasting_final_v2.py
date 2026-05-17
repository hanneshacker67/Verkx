
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── Paths ──────────────────────────────────────────────────────────────────────
VERKX      = "/mnt/c/Users/Hanne/Documents/verkx"
DAS_CSV    = f"{VERKX}/das_features.csv"
WAM_PKL    = f"{VERKX}/storm_oct_2024/wave_data_fullset_100km_oct2024.pkl"
WIND_CSV   = f"{VERKX}/vindgogn_01.10-13.10.csv"
OUTPUT_DIR = f"{VERKX}/pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
WAM_SITE_KM = 9.2
TRAIN_DAYS  = [datetime.date(2024, 10, d) for d in [1, 2, 3, 5, 8, 9, 12]]
TEST_DAYS   = [datetime.date(2024, 10, d) for d in [4, 6, 10, 11, 13]]
HORIZONS    = [0, 1, 2, 3, 6, 12, 24]


WIND_LAGS = list(range(6, 21))


def make_model(h):
    if h <= 3:
        return SVR(kernel='rbf', C=5, epsilon=0.03), "SVR"
    else:
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=3,
            learning_rate=0.05, random_state=42
        ), "GradientBoosting"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")

das = pd.read_csv(DAS_CSV)
das["timestamp"] = pd.to_datetime(das["timestamp"], utc=True).dt.tz_localize(None)
das["hour_utc"]  = das["timestamp"].dt.floor("h")
das["date"]      = das["timestamp"].dt.date
das_h = das.groupby("hour_utc").agg(
    rms=("rms", "mean"),
    variance=("variance", "mean"),
    wave_band_energy=("wave_band_energy", "mean"),
    shore_power=("shore_power", "mean"),
    bandpass_rms=("bandpass_rms", "mean"),
    temporal_variability=("temporal_variability", "mean"),
    dominant_period_s=("dominant_period_s", "mean"),
    date=("date", "first"),
).reset_index()

with open(WAM_PKL, "rb") as f:
    wam_raw = pickle.load(f)
pt_idx = int(np.argmin(np.abs(wam_raw["dist"] - WAM_SITE_KM)))
wam_df = pd.DataFrame({"hour_utc": pd.to_datetime(wam_raw["time"])})
wam_df["VHM0"] = wam_raw["VHM0"][:, pt_idx]  # target only — no other WAM vars

wind_raw = pd.read_csv(WIND_CSV, parse_dates=["time"])
wind_raw["time"]     = pd.to_datetime(wind_raw["time"], utc=True).dt.tz_localize(None)
wind_raw["hour_utc"] = wind_raw["time"].dt.floor("h")
wind_h = wind_raw.groupby("hour_utc").agg(
    wind_speed=("f",  "mean"),
    wind_gust =("fg", "max"),
    wind_sin  =("d",  lambda x: np.sin(np.radians(x)).mean()),
    wind_cos  =("d",  lambda x: np.cos(np.radians(x)).mean()),
).reset_index()

df = (das_h
      .merge(wam_df, on="hour_utc", how="inner")
      .merge(wind_h, on="hour_utc", how="left")
      .sort_values("hour_utc")
      .reset_index(drop=True))
df["date"] = df["hour_utc"].dt.date
print(f"  {len(df)} hourly records")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("Building features...")

DAS_BASE  = ["rms", "variance", "wave_band_energy", "shore_power",
             "bandpass_rms", "temporal_variability", "dominant_period_s"]
WIND_BASE = ["wind_speed", "wind_gust", "wind_sin", "wind_cos"]

feat_cols = {}

# ── DAS features ────────────────────────────────────────────────────────────
for col in DAS_BASE:
    # Lag features
    for lag in [1, 2, 3, 6, 12]:
        feat_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    # Rolling mean and std
    for w in [3, 6, 12]:
        feat_cols[f"{col}_roll{w}"]     = df[col].shift(1).rolling(w).mean()
        feat_cols[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w).std()
    # Trend: rate of change
    for t in [3, 6, 12]:
        feat_cols[f"{col}_trend{t}"] = df[col].shift(1) - df[col].shift(t)

# ── Acceleration features ────────────────────────────────────────────────────

for col in DAS_BASE:
    trend_now  = df[col] - df[col].shift(1)
    trend_prev = df[col].shift(1) - df[col].shift(2)
    feat_cols[f"{col}_accel"] = trend_now - trend_prev

    trend_3h      = df[col] - df[col].shift(3)
    trend_3h_prev = df[col].shift(3) - df[col].shift(6)
    feat_cols[f"{col}_accel3h"] = trend_3h - trend_3h_prev

# Wind acceleration
wind_trend_now  = df["wind_speed"] - df["wind_speed"].shift(1)
wind_trend_prev = df["wind_speed"].shift(1) - df["wind_speed"].shift(2)
feat_cols["wind_accel"] = wind_trend_now - wind_trend_prev

wind_trend_3h      = df["wind_speed"] - df["wind_speed"].shift(3)
wind_trend_3h_prev = df["wind_speed"].shift(3) - df["wind_speed"].shift(6)
feat_cols["wind_accel3h"] = wind_trend_3h - wind_trend_3h_prev

# ── Wind lag features (data-driven: 6-20h) ──────────────────────────────────

for col in WIND_BASE:
    for lag in WIND_LAGS:
        feat_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
  
    for w in [6, 12]:
        feat_cols[f"{col}_roll_gen{w}"] = df[col].shift(6).rolling(w).mean()

df = pd.concat([df, pd.DataFrame(feat_cols, index=df.index)], axis=1)
df = df.dropna().reset_index(drop=True)
df["date"] = df["hour_utc"].dt.date

# All features
FEATURES = [c for c in df.columns if c not in ["hour_utc", "date", "VHM0"]]
print(f"  Total features: {len(FEATURES)}")

print(f"  Wind lags {WIND_LAGS[0]}-{WIND_LAGS[-1]}h (data-driven)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN AND EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nTraining for horizons: {HORIZONS}")

results = {}

for h in HORIZONS:
    target = df["VHM0"].shift(-h)
    valid  = target.notna()
    df2    = df[valid].copy()
    df2["target"] = target[valid].values

    train = df2[df2["date"].isin(TRAIN_DAYS)]
    test  = df2[df2["date"].isin(TEST_DAYS)]

    # Naive baseline
    naive_r2 = r2_score(
        test["target"],
        np.full(len(test), train["target"].mean())
    )

    sc   = StandardScaler()
    X_tr = sc.fit_transform(train[FEATURES].values)
    X_te = sc.transform(test[FEATURES].values)

    mdl, mdl_name = make_model(h)
    mdl.fit(X_tr, train["target"])
    pred = np.clip(mdl.predict(X_te), 0.3, 2.5)

    r2   = r2_score(test["target"].values, pred)
    rmse = np.sqrt(mean_squared_error(test["target"].values, pred))

    results[h] = {
        "r2": r2, "rmse": rmse, "model": mdl_name,
        "naive_r2": naive_r2,
        "pred": pred,
        "true": test["target"].values,
        "time": test["hour_utc"].values,
    }
    print(f"  h={h:2d}h  R²={r2:.3f}  RMSE={rmse:.3f}m  [{mdl_name}]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
v1_r2 = {0:0.980, 1:0.963, 2:0.929, 3:0.857, 6:0.779, 12:0.585, 24:-0.325}

print(f"\n{'='*70}")
print("FINAL RESULTS: DAS + Wind (lags 6-20h) + Acceleration → WAM Hs")
print(f"{'='*70}")
print(f"{'Horizon':10} {'R²':8} {'RMSE':10} {'Model':22} {'v1 R²':8}")
for h in HORIZONS:
    r    = results[h]
    diff = r["r2"] - v1_r2[h]
    
    print(f"h={h:2d}h      {r['r2']:.3f}   {r['rmse']:.3f}m     "
          f"{r['model']:22s} {v1_r2[h]:+.3f}   ")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOT — R² vs horizon
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#F8F9FA")

r2_new   = [results[h]["r2"]       for h in HORIZONS]
r2_naive = [results[h]["naive_r2"] for h in HORIZONS]
r2_v1    = [v1_r2[h]              for h in HORIZONS]
rmse_new = [results[h]["rmse"]     for h in HORIZONS]

ax = axes[0]
ax.plot(HORIZONS, r2_naive, "o:", color="gray",    lw=1.5, ms=6,
        label="Naive baseline (training mean)")
ax.plot(HORIZONS, r2_v1,   "o--", color="tomato",  lw=2,   ms=8,
        label="v1 — DAS+Wind+WAM inputs (soft leakage)")
ax.plot(HORIZONS, r2_new,  "o-",  color="#2471A3", lw=2.5, ms=8,
        label="v2 — DAS+Wind only (honest) ✓")
ax.axhline(0, color="gray", ls=":", lw=1)
ax.set_xlabel("Forecast horizon (hours)", fontsize=12)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² vs forecast horizon\nDAS + Wind → WAM Hs",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_xticks(HORIZONS); ax.set_ylim(-0.5, 1.05)

ax = axes[1]
ax.plot(HORIZONS, rmse_new, "o-", color="#2471A3", lw=2.5, ms=8)
for h, r2, rmse in zip(HORIZONS, r2_new, rmse_new):
    ax.annotate(f"R²={r2:.2f}", (h, rmse), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color="#2471A3")
ax.set_xlabel("Forecast horizon (hours)", fontsize=12)
ax.set_ylabel("RMSE (m)", fontsize=12)
ax.set_title("RMSE vs forecast horizon\nDAS + Wind → WAM Hs",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3); ax.set_ylim(bottom=0); ax.set_xticks(HORIZONS)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "forecasting_final_v2.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PLOT — Time series at h=0, h=6, h=12, h=24
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)
fig.patch.set_facecolor("#F8F9FA")

colors = ["#1E8449", "#2471A3", "#8E44AD", "#C0392B"]
for ax, h, col in zip(axes, [0, 6, 12, 24], colors):
    r = results[h]
    t = pd.to_datetime(r["time"])
    ax.plot(t, r["true"], color="tomato", lw=2, label="WAM Hs (actual)")
    ax.plot(t, r["pred"], "--", color=col, lw=2,
            label=f"h={h}h forecast   R²={r['r2']:.3f}, RMSE={r['rmse']:.3f}m")
    ax.set_ylabel("Hs (m)", fontsize=11)
    ax.set_title(f"h={h}h — {r['model']}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())

fig.suptitle(
    "Wave Height Forecasting — DAS + Wind (lags 6-20h) + Acceleration\n"
    "Test days: Oct 4, 6, 10, 11, 13  ·  Input: DAS + Wind only  ·  Target: WAM Hs",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "forecasting_timeseries_final_v2.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")
print("\nDone!")
