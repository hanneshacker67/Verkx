
import os, pickle, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
VERKX      = "/mnt/c/Users/Hanne/Documents/verkx"
DAS_CSV    = f"{VERKX}/das_features.csv"
WAM_PKL    = f"{VERKX}/storm_oct_2024/wave_data_fullset_100km_oct2024.pkl"
WIND_CSV   = f"{VERKX}/vindgogn_01.10-13.10.csv"
OUTPUT_DIR = f"{VERKX}/pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DAYS = [datetime.date(2024,10,d) for d in [1,2,3,5,8,9,12]]
TEST_DAYS  = [datetime.date(2024,10,d) for d in [4,6,10,11,13]]
WIND_LAGS  = list(range(6, 21))
HORIZON    = 6  

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
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

with open(WAM_PKL,"rb") as f:
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("Building features...")
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

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN & PREDICT
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

target = df["VHM0"].shift(-HORIZON)
valid  = target.notna()
df2    = df[valid].copy()
df2["target"] = target[valid].values

tr = df2[df2["date"].isin(TRAIN_DAYS)]
te = df2[df2["date"].isin(TEST_DAYS)]

sc   = StandardScaler()
X_tr = sc.fit_transform(tr[FEATS].values)
X_te = sc.transform(te[FEATS].values)

mdl = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
mdl.fit(X_tr, tr["target"])
pred = np.clip(mdl.predict(X_te), 0.3, 2.5)

r2   = r2_score(te["target"].values, pred)
rmse = np.sqrt(mean_squared_error(te["target"].values, pred))
print(f"  h={HORIZON}h: R²={r2:.3f}, RMSE={rmse:.3f}m")


all_test_rows = df[df["date"].isin(TEST_DAYS)].copy()
X_all_te = sc.transform(all_test_rows[FEATS].values)
all_test_rows["pred"] = np.clip(mdl.predict(X_all_te), 0.3, 2.5)

all_wam = wam_df[
    (wam_df["hour_utc"] >= df["hour_utc"].min()) &
    (wam_df["hour_utc"] <= df["hour_utc"].max())
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────────────────────────────────────
BG    = "#0B1829"
CYAN  = "#00D4E8"
GREEN = "#4ADE80"
GRID  = "#162840"
WHITE = "#FFFFFF"
LGRAY = "#8899BB"


TEST_INFO = [
    {"date": datetime.date(2024,10,4),  "label": "Oct 4 — Stormy", "col": "#E74C3C", "bg": "#3D0A0A"},
    {"date": datetime.date(2024,10,6),  "label": "Oct 6 — Stormy", "col": "#E74C3C", "bg": "#3D0A0A"},
    {"date": datetime.date(2024,10,10), "label": "Oct 10 — Swell", "col": "#F39C12", "bg": "#3D2A05"},
    {"date": datetime.date(2024,10,11), "label": "Oct 11 — Calm",  "col": "#27AE60", "bg": "#0A2D18"},
    {"date": datetime.date(2024,10,13), "label": "Oct 13 — Calm",  "col": "#27AE60", "bg": "#0A2D18"},
]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.12)

# Shade test days
for ti in TEST_INFO:
    start = pd.Timestamp(ti["date"])
    end   = start + pd.Timedelta(hours=23, minutes=59)
    ax.axvspan(start, end, color=ti["bg"], alpha=0.9, zorder=1)
    ax.axvline(start, color=ti["col"], lw=1.0, ls="--", alpha=0.6, zorder=2)

# Grid
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
ax.xaxis.grid(False)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.spines["left"].set_color(GRID)

# Lines
ax.plot(all_wam["hour_utc"], all_wam["VHM0"],
        color=CYAN, lw=2.0, label="WAM Hₛ (actual)", zorder=4)
# Plot each test day separately — lines don't connect over training days
first_label = True
for d in sorted(TEST_DAYS):
    day_data = all_test_rows[all_test_rows["date"] == d].sort_values("hour_utc")
    if len(day_data) == 0:
        continue
    label = f"Gradient Boosting {HORIZON}h forecast (test days)" if first_label else None
    ax.plot(day_data["hour_utc"], day_data["pred"],
            color=GREEN, lw=1.5, ls="--", alpha=0.9,
            label=label, zorder=5)
    first_label = False

# Test day labels
for ti in TEST_INFO:
    x    = pd.Timestamp(ti["date"]) + pd.Timedelta(hours=1.5)
    vals = all_wam[all_wam["hour_utc"].dt.date == ti["date"]]["VHM0"]
    y    = vals.max() + 0.06 if len(vals) > 0 else 1.5
    ax.text(x, y, ti["label"],
            color=WHITE, fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=ti["col"],
                      alpha=0.85, edgecolor="none"),
            zorder=8)

# Titles
fig.text(0.06, 0.95,
         f"{HORIZON}-hour forecast on held-out test days",
         color=WHITE, fontsize=22, fontweight="bold", va="top")
fig.text(0.06, 0.895,
         f"Gradient Boosting  ·  DAS + wind lags 6–20h  ·  "
         f"no WAM inputs  ·  R²={r2:.3f},  RMSE={rmse:.3f} m",
         color=CYAN, fontsize=13, fontstyle="italic", va="top", alpha=0.85)

# Watermark
ax.text(0.99, 0.04,
        f"R² = {r2:.3f}  ·  RMSE = {rmse:.3f} m  ·  "
        f"DAS + wind lags 6–20h  ·  no WAM inputs",
        transform=ax.transAxes, color=LGRAY, fontsize=9,
        ha="right", va="bottom", alpha=0.7)

ax.tick_params(colors=LGRAY, labelsize=11)
ax.set_ylabel("Hₛ (m)", color=LGRAY, fontsize=13, labelpad=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.set_ylim(bottom=0.2)

ax.legend(fontsize=11, facecolor="#0D2035", edgecolor=GRID,
          labelcolor=WHITE, loc="upper right",
          framealpha=0.9, borderpad=0.8)

outpath = os.path.join(OUTPUT_DIR, "timeseries_6h_styled.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"\nSaved → {outpath}")
print("Done!")
