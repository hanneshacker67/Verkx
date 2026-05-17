

import os, pickle, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
VERKX      = "/mnt/c/Users/Hanne/Documents/verkx"
DAS_CSV    = f"{VERKX}/das_features.csv"
WAM_PKL    = f"{VERKX}/storm_oct_2024/wave_data_fullset_100km_oct2024.pkl"
WIND_CSV   = f"{VERKX}/vindgogn_01.10-13.10.csv"
OUTPUT_DIR = f"{VERKX}/pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAUNCH_TIME = pd.Timestamp("2024-10-10 06:00:00")
TRAIN_DAYS  = [datetime.date(2024,10,d) for d in [1,2,3,5,8,9,12]]
TEST_DAYS   = [datetime.date(2024,10,d) for d in [4,6,10,11,13]]
WIND_LAGS   = list(range(6, 21))
N_BOOTSTRAP = 50

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
das = pd.read_csv(DAS_CSV)
das["timestamp"] = pd.to_datetime(das["timestamp"], utc=True).dt.tz_localize(None)
das["hour_utc"]  = das["timestamp"].dt.floor("h")
das["date"]      = das["timestamp"].dt.date
das_h = das.groupby("hour_utc").agg(
    rms=("rms","mean"), variance=("variance","mean"),
    wave_band_energy=("wave_band_energy","mean"),
    shore_power=("shore_power","mean"),
    bandpass_rms=("bandpass_rms","mean"),
    temporal_variability=("temporal_variability","mean"),
    dominant_period_s=("dominant_period_s","mean"),
    date=("date","first"),
).reset_index()

with open(WAM_PKL, "rb") as f:
    wam_raw = pickle.load(f)
pt_idx = int(np.argmin(np.abs(wam_raw["dist"] - 9.2)))
wam_df = pd.DataFrame({"hour_utc": pd.to_datetime(wam_raw["time"])})
wam_df["VHM0"] = wam_raw["VHM0"][:, pt_idx]

wind_raw = pd.read_csv(WIND_CSV, parse_dates=["time"])
wind_raw["time"]     = pd.to_datetime(wind_raw["time"], utc=True).dt.tz_localize(None)
wind_raw["hour_utc"] = wind_raw["time"].dt.floor("h")
wind_h = wind_raw.groupby("hour_utc").agg(
    wind_speed=("f","mean"), wind_gust=("fg","max"),
    wind_sin=("d", lambda x: np.sin(np.radians(x)).mean()),
    wind_cos=("d", lambda x: np.cos(np.radians(x)).mean()),
).reset_index()

df = (das_h.merge(wam_df, on="hour_utc", how="inner")
          .merge(wind_h, on="hour_utc", how="left")
          .sort_values("hour_utc").reset_index(drop=True))
df["date"] = df["hour_utc"].dt.date

# ── Features ───────────────────────────────────────────────────────────────────
DAS_BASE  = ["rms","variance","wave_band_energy","shore_power",
             "bandpass_rms","temporal_variability","dominant_period_s"]
WIND_BASE = ["wind_speed","wind_gust","wind_sin","wind_cos"]

feat_cols = {}
for col in DAS_BASE:
    for lag in [1,2,3,6,12]:
        feat_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    for w in [3,6,12]:
        feat_cols[f"{col}_roll{w}"]     = df[col].shift(1).rolling(w).mean()
        feat_cols[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w).std()
    for t in [3,6,12]:
        feat_cols[f"{col}_trend{t}"] = df[col].shift(1) - df[col].shift(t)
    feat_cols[f"{col}_accel"] = (
        (df[col] - df[col].shift(1)) - (df[col].shift(1) - df[col].shift(2))
    )

for col in WIND_BASE:
    for lag in WIND_LAGS:
        feat_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    for w in [6,12]:
        feat_cols[f"{col}_roll{w}"] = df[col].shift(6).rolling(w).mean()

df = pd.concat([df, pd.DataFrame(feat_cols, index=df.index)], axis=1)
df = df.dropna().reset_index(drop=True)
df["date"] = df["hour_utc"].dt.date

FEATS = [c for c in df.columns if c not in ["hour_utc","date","VHM0"]]
print(f"  Features: {len(FEATS)}")

launch_mask = df["hour_utc"] == LAUNCH_TIME
assert launch_mask.sum() > 0, f"Launch time {LAUNCH_TIME} not found!"

def make_model(h):
    if h <= 3:
        return SVR(kernel="rbf", C=5, epsilon=0.03)
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)

# ── Train & forecast ───────────────────────────────────────────────────────────
print(f"\nTraining 24 models with {N_BOOTSTRAP} bootstrap samples each")
print(f"Launch: {LAUNCH_TIME}")

forecast_times = []
pred_mean      = []
pred_lower     = []
pred_upper     = []
wam_vals       = []
rmse_by_h      = []

for h in range(1, 25):
    target = df["VHM0"].shift(-h)
    valid  = target.notna()
    df2    = df[valid].copy()
    df2["target"] = target[valid].values

    tr = df2[df2["date"].isin(TRAIN_DAYS)]
    te = df2[df2["date"].isin(TEST_DAYS)]

    sc   = StandardScaler()
    X_tr = sc.fit_transform(tr[FEATS].values)
    X_te = sc.transform(te[FEATS].values)

    mdl = make_model(h)
    mdl.fit(X_tr, tr["target"])

    y_pred_te = np.clip(mdl.predict(X_te), 0.3, 2.5)
    rmse = float(np.sqrt(mean_squared_error(te["target"].values, y_pred_te)))

    # Launch prediction
    X_launch = sc.transform(df.loc[launch_mask, FEATS].values)
    y_launch = float(np.clip(mdl.predict(X_launch)[0], 0.3, 2.5))

    # Bootstrap confidence interval
    rng = np.random.RandomState(42 + h)
    n_tr = len(tr)
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx  = rng.choice(n_tr, size=n_tr, replace=True)
        m2   = make_model(h)
        sc2  = StandardScaler()
        X_b  = sc2.fit_transform(tr[FEATS].values[idx])
        m2.fit(X_b, tr["target"].values[idx])
        X_l2 = sc2.transform(df.loc[launch_mask, FEATS].values)
        boots.append(float(np.clip(m2.predict(X_l2)[0], 0.3, 2.5)))

    pred_time = LAUNCH_TIME + pd.Timedelta(hours=h)
    wam_at    = wam_df[wam_df["hour_utc"] == pred_time]["VHM0"]
    wam_val   = float(wam_at.values[0]) if len(wam_at) > 0 else np.nan

    forecast_times.append(pred_time)
    pred_mean.append(y_launch)
    pred_lower.append(float(np.percentile(boots, 10)))
    pred_upper.append(float(np.percentile(boots, 90)))
    wam_vals.append(wam_val)
    rmse_by_h.append(rmse)

    if h in [1, 6, 12, 24]:
        print(f"  h={h:2d}: pred={y_launch:.3f}m  CI=[{pred_lower[-1]:.3f},{pred_upper[-1]:.3f}]"
              f"  WAM={wam_val:.3f}m  RMSE={rmse:.3f}m")

fdf = pd.DataFrame({
    "pred_time": forecast_times,
    "pred_mean": pred_mean,
    "pred_lower": pred_lower,
    "pred_upper": pred_upper,
    "wam": wam_vals,
    "rmse": rmse_by_h,
    "h": list(range(1, 25)),
})

wam_plot = wam_df[
    (wam_df["hour_utc"] >= LAUNCH_TIME) &
    (wam_df["hour_utc"] <= LAUNCH_TIME + pd.Timedelta(hours=25))
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 7))
                                
fig.patch.set_facecolor("white")

# WAM actual
ax1.plot(wam_plot["hour_utc"], wam_plot["VHM0"],
         color="tomato", lw=2.5, label="WAM Hs (actual)", zorder=5)

# Confidence interval
ax1.fill_between(fdf["pred_time"], fdf["pred_lower"], fdf["pred_upper"],
                  alpha=0.25, color="#8E44AD",
                  label="10%–90% confidence interval", zorder=3)

# Coloured forecast line (green=short, red=long)
cmap = plt.cm.RdYlGn_r
for i in range(len(fdf) - 1):
    ax1.plot(
        [fdf["pred_time"].iloc[i], fdf["pred_time"].iloc[i+1]],
        [fdf["pred_mean"].iloc[i], fdf["pred_mean"].iloc[i+1]],
        color=cmap(fdf["h"].iloc[i] / 24), lw=2.5, zorder=4,
    )

# Key horizon markers
for h_mark in [6, 12, 24]:
    row = fdf[fdf["h"] == h_mark].iloc[0]
    ax1.scatter(row["pred_time"], row["pred_mean"],
                color=cmap(h_mark / 24), s=130, zorder=6,
                label=f"h={h_mark}h: {row['pred_mean']:.2f}m  (WAM: {row['wam']:.2f}m)")

ax1.axvline(LAUNCH_TIME, color="black", lw=1.5, ls="--", alpha=0.5)
ax1.text(LAUNCH_TIME + pd.Timedelta(minutes=35),
         wam_plot["VHM0"].max() * 0.99,
         f"Launch\n{LAUNCH_TIME.strftime('%H:%M')}",
         fontsize=9, va="top")

ax1.set_title(
    f"Single Launch Forecast — Oct 4\n"
    f"Launch: {LAUNCH_TIME.strftime('%-d. okt %H:%M UTC')}  ·  "
    f"h=1 to h=24  ·  10%–90% confidence interval (bootstrap, n={N_BOOTSTRAP})\n"
    f"Models: SVR (h≤3h) / Gradient Boosting (h>3h)  ·  Input: DAS + Wind",
    fontsize=11, fontweight="bold",
)
ax1.set_ylabel("Hs (m)", fontsize=11)
ax1.legend(fontsize=9, loc="lower left")
ax1.grid(alpha=0.3)
ax1.set_ylim(bottom=0)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d. okt"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=3))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 24))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, pad=0.01, shrink=0.8)
cbar.set_label("Forecast horizon (h)", fontsize=9)



plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, "single_launch_forecast_v2_10.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {outpath}")
print("Done!")
