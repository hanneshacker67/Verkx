

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

# ── Paths ──────────────────────────────────────────────────────────────────────
VERKX      = "/mnt/c/Users/Hanne/Documents/verkx"
DAS_CSV    = f"{VERKX}/das_features.csv"
WAM_PKL    = f"{VERKX}/storm_oct_2024/wave_data_fullset_100km_oct2024.pkl"
WIND_CSV   = f"{VERKX}/vindgogn_01.10-13.10.csv"
OUTPUT_DIR = f"{VERKX}/pipeline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
WAM_SITE_KM  = 9.2
TRAIN_DAYS   = [datetime.date(2024, 10, d) for d in [1, 2, 3, 5, 8, 9, 12]]
TEST_DAYS    = [datetime.date(2024, 10, d) for d in [4, 6, 10, 11, 13]]
N_CLUSTERS   = 3
N_TOP_FEATS  = 15
N_ESTIMATORS = 500

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
wam_df["VHM0"] = wam_raw["VHM0"][:, pt_idx]  # target only

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


for col in DAS_BASE:
    for lag in [1, 2, 3, 6, 12]:
        feat_cols[f"{col}_lag{lag}"] = df[col].shift(lag)
    for w in [3, 6, 12]:
        feat_cols[f"{col}_roll{w}"]     = df[col].shift(1).rolling(w).mean()
        feat_cols[f"{col}_roll{w}_std"] = df[col].shift(1).rolling(w).std()



df = pd.concat([df, pd.DataFrame(feat_cols, index=df.index)], axis=1)
df = df.dropna().reset_index(drop=True)
df["date"] = df["hour_utc"].dt.date


CANDIDATES = [
    c for c in df.columns
    if c not in ["hour_utc", "date", "VHM0", "label"]
]
print(f"  Candidate features: {len(CANDIDATES)}")
print(f"    DAS raw:              {len(DAS_BASE)}")
print(f"    DAS lag (1,2,3,6,12): {len(DAS_BASE) * 5}")
print(f"    DAS rolling (3,6,12): {len(DAS_BASE) * 3 * 2}")
print(f"    Wind raw:             {len(WIND_BASE)}")
assert not any(f in CANDIDATES for f in ["VTPK", "VMDR", "VTM10", "VHM0_WW"]), \
    "WAM leakage detected!"
print(f"  ✓ No WAM features in candidates")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train = df[df["date"].isin(TRAIN_DAYS)].copy()
test  = df[df["date"].isin(TEST_DAYS)].copy()
print(f"  Train: {len(train)} samples | Test: {len(test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# 4. THRESHOLDS — K-MEANS 
# ─────────────────────────────────────────────────────────────────────────────
print("\nFitting k-means thresholds on training data...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km.fit(train[["VHM0"]])
centers = sorted(km.cluster_centers_.ravel())
T1 = float((centers[0] + centers[1]) / 2)
T2 = float((centers[1] + centers[2]) / 2)
print(f"  Cluster centers: {[f'{c:.3f}m' for c in centers]}")
print(f"  → calm  < {T1:.3f} m")
print(f"  → swell   {T1:.3f} – {T2:.3f} m")
print(f"  → stormy ≥ {T2:.3f} m")

def label_hs(hs):
    if hs < T1:  return "calm"
    if hs < T2:  return "swell"
    return "stormy"

train["label"] = train["VHM0"].apply(label_hs)
test["label"]  = test["VHM0"].apply(label_hs)

print(f"\n  Train label distribution:")
print(train["label"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE SELECTION 
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSelecting top {N_TOP_FEATS} features on training data...")
sc_sel = StandardScaler()
X_all  = sc_sel.fit_transform(train[CANDIDATES].values)

et_sel = ExtraTreesClassifier(
    n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1,
    class_weight={"calm": 1, "swell": 1.5, "stormy": 2},
)
et_sel.fit(X_all, train["label"])

imp = pd.Series(et_sel.feature_importances_, index=CANDIDATES).sort_values(ascending=False)
TOP_FEATURES = list(imp.head(N_TOP_FEATS).index)

print(f"  Top {N_TOP_FEATS} features:")
for i, f in enumerate(TOP_FEATURES):
    kind = "WIND" if "wind" in f else "DAS"
    print(f"    {i+1:2d}. [{kind}] {f}  ({imp[f]:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nTraining Random Forest...")
sc = StandardScaler()
X_tr = sc.fit_transform(train[TOP_FEATURES].values)
X_te = sc.transform(test[TOP_FEATURES].values)

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1,
    class_weight={"calm": 1, "swell": 1, "stormy": 2},
)
clf.fit(X_tr, train["label"])

# ─────────────────────────────────────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_te)
y_true = test["label"].values

acc = accuracy_score(y_true, y_pred)
rep = classification_report(y_true, y_pred, zero_division=0)
rep_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
cm  = confusion_matrix(y_true, y_pred, labels=["calm", "swell", "stormy"])

print(f"\n{'='*60}")
print(f"RESULTS — test set (Oct 4, 6, 10, 11, 13)")
print(f"{'='*60}")
print(f"Input:     DAS + Wind")
print(f"Features:  {N_TOP_FEATS} selected from {len(CANDIDATES)} candidates")
print(f"Threshold: calm<{T1:.3f}m, swell<{T2:.3f}m, stormy≥{T2:.3f}m (k-means training)")
print(f"\n{rep}")
print(f"Accuracy: {acc:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PLOT
# ─────────────────────────────────────────────────────────────────────────────
CLASSES  = ["calm", "swell", "stormy"]
CCOLORS  = {"calm": "#1E8449", "swell": "#2471A3", "stormy": "#C0392B"}

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#F8F9FA")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── Confusion matrix ──────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        color = "white" if cm_norm[i, j] > 0.55 else "#1a1a1a"
        ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)
ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_yticklabels(CLASSES, fontsize=11)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title(f"Random Forest\nAccuracy: {acc*100:.1f}%",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8)

# ── Per-class metrics ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
metrics = ["precision", "recall", "f1-score"]
bar_colors = ["#2E86AB", "#A23B72", "#F18F01"]
x = np.arange(3)
width = 0.25
for mi, metric in enumerate(metrics):
    vals = [rep_dict.get(c, {}).get(metric, 0) for c in CLASSES]
    bars = ax.bar(x + mi * width, vals, width,
                  label=metric.replace("-score", ""),
                  color=bar_colors[mi], alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Precision / Recall / F1\nper class",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.9, color="gray", ls="--", lw=1, alpha=0.5)
ax.grid(axis="y", alpha=0.3)

# ── Feature importance ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
fi = pd.Series(clf.feature_importances_, index=TOP_FEATURES).sort_values()
fi_colors = ["#27AE60" if "wind" in f else "#2980B9" for f in fi.index]
ax.barh(range(len(fi)), fi.values, color=fi_colors, alpha=0.85)
ax.set_yticks(range(len(fi)))
ax.set_yticklabels(fi.index, fontsize=8)
ax.set_xlabel("Importance", fontsize=11)
ax.set_title(f"Feature Importance\n(top {N_TOP_FEATS})",
             fontsize=12, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#2980B9", label="DAS features"),
    mpatches.Patch(color="#27AE60", label="Wind features"),
], fontsize=9, loc="lower right")
ax.grid(axis="x", alpha=0.3)

# ── Time series ───────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :])
test_plot = test.copy()
test_plot["pred"] = y_pred
test_plot = test_plot.sort_values("hour_utc").reset_index(drop=True)

ax.plot(test_plot["hour_utc"], test_plot["VHM0"],
        color="tomato", lw=2.5, label="WAM Hs (m)", zorder=5)

prev_x = test_plot["hour_utc"].iloc[0]
prev_l = test_plot["label"].iloc[0]
for i in range(1, len(test_plot)):
    if test_plot["label"].iloc[i] != prev_l or i == len(test_plot) - 1:
        ax.axvspan(prev_x, test_plot["hour_utc"].iloc[i],
                   alpha=0.12, color=CCOLORS[prev_l], zorder=1)
        prev_x = test_plot["hour_utc"].iloc[i]
        prev_l = test_plot["label"].iloc[i]

wrong = test_plot[test_plot["label"] != test_plot["pred"]]
ax.scatter(wrong["hour_utc"], wrong["VHM0"],
           marker="x", s=140, color="black", lw=2.5, zorder=6,
           label=f"Wrong prediction ({len(wrong)})")

ax.axhline(T1, color=CCOLORS["calm"],   ls="--", lw=1.2, alpha=0.7,
           label=f"calm/swell {T1:.3f}m")
ax.axhline(T2, color=CCOLORS["stormy"], ls="--", lw=1.2, alpha=0.7,
           label=f"swell/stormy {T2:.3f}m")

import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set_xlabel("Date (UTC)", fontsize=11)
ax.set_ylabel("Hs (m)", fontsize=11)
ax.set_title(
    f"Test set — WAM Hs with actual condition shading  |  "
    f"{len(wrong)}/{len(test_plot)} wrong ({100*len(wrong)/len(test_plot):.1f}% error)",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=9, ncol=5)
ax.grid(alpha=0.3)

fig.suptitle(
    f"Wave Condition Classification — Random Forest\n"
    f"Input: DAS + Wind  ·  Top {N_TOP_FEATS} of {len(CANDIDATES)} features  ·  "
    f"K-means thresholds (training only)  ·  Accuracy: {acc*100:.1f}%",
    fontsize=12, fontweight="bold", y=0.98
)

outpath = os.path.join(OUTPUT_DIR, "classification_results_v3.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {outpath}")
print("Done!")
