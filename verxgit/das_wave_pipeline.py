# NOTE: Steps 5 and 6 (classifier and regressor) in this script are 
# exploratory only and do not produce the results reported in the paper.
# Final models are in classification_final_v3.py and forecasting_final_v2.py

import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy import signal
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DAS_DIR          = "/mnt/c/Users/Hanne/Documents/verkx/datafiles"
WAVE_MODEL_PATH  = "/mnt/c/Users/Hanne/Documents/verkx/storm_oct_2024/wave_data_100km_oct2024.pkl"
WAVE_MODEL_DIST_KM = 9.2   

ACTIVE_MIN_M = 10_000       
ACTIVE_MAX_M = 11_500

# Equal-thirds thresholds from exploration (only used for exploration not in the regression or classification models)
HS_CALM_MAX   = 0.75
HS_STORMY_MIN = 1.10


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — WAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_wave_model(path, dist_km):
    with open(path, "rb") as f:
        ds = pickle.load(f)
    dist   = np.array(ds["dist"])
    hs_all = np.array(ds["wave_hs"])
    times  = np.array(ds["time"])
    col    = np.argmin(np.abs(dist - dist_km))
    print(f"Wave model: dist={dist[col]:.2f} km | "
          f"Hs {np.nanmin(hs_all[:,col]):.2f}-{np.nanmax(hs_all[:,col]):.2f} m")
    lookup = {}
    for i, t in enumerate(times):
        ts = pd.Timestamp(str(t)).tz_localize("UTC")
        lookup[ts] = float(hs_all[i, col])
    return lookup


def get_hs(lookup, timestamp):
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return lookup.get(ts.round("h"), np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(xx, tt, data):
    mask   = (xx >= ACTIVE_MIN_M) & (xx <= ACTIVE_MAX_M)
    active = data[:, mask]
    xx_act = xx[mask]
    dt     = float(np.median(np.diff(tt)))
    fs     = 1.0 / dt

    # 1. Variance
    variance = float(np.mean(np.var(active, axis=0)))

    # 2. RMS
    rms = float(np.sqrt(np.mean(active ** 2)))

    # 3 & 4. Dominant period and wave-band energy
    nperseg = min(256, active.shape[0] // 2)
    freqs, psd = signal.welch(active, fs=fs, axis=0, nperseg=nperseg)
    psd_mean   = psd.mean(axis=1)
    wave_band  = (freqs >= 0.05) & (freqs <= 0.5)
    if wave_band.any():
        psd_wb           = psd_mean[wave_band]
        peak_freq        = freqs[wave_band][np.argmax(psd_wb)]
        dominant_period  = float(1.0 / peak_freq) if peak_freq > 0 else np.nan
        wave_band_energy = float(np.trapezoid(psd_wb, freqs[wave_band]))
    else:
        dominant_period  = np.nan
        wave_band_energy = 0.0

    # 5. Near-shore power
    shore_mask  = xx_act >= (xx_act.max() - 500)
    shore_power = float(np.mean(np.var(active[:, shore_mask], axis=0))) \
                  if shore_mask.any() else variance

    # 6. Phase velocity
    dx = float(np.median(np.diff(xx_act)))
    velocities = []
    for i in range(min(30, active.shape[1] - 1)):
        xcorr = np.correlate(active[:, i], active[:, i+1], mode="full")
        lags  = np.arange(-(len(tt)-1), len(tt))
        lag   = int(lags[np.argmax(np.abs(xcorr))])
        if lag != 0:
            v = dx / (lag * dt)
            if 2 < abs(v) < 40:
                velocities.append(abs(v))
    phase_velocity = float(np.median(velocities)) if velocities else np.nan

    # 7. Spatial coherence
    coh = [abs(np.corrcoef(active[:, i], active[:, i+1])[0, 1])
           for i in range(min(40, active.shape[1] - 1))]
    spatial_coherence = float(np.mean(coh)) if coh else np.nan

    # 8. Temporal variability
    channel_rms          = np.sqrt(np.mean(active ** 2, axis=0))
    temporal_variability = float(np.std(channel_rms))

    # 9. Bandpass RMS (NEW) — ocean wave band only, spatially averaged
    #    Removes instrument noise, temperature drift, microseisms
    #    Keeps only 0.05-0.5 Hz where ocean wave energy lives
    try:
        spatial_avg = active.mean(axis=1)
        sos         = signal.butter(4, [0.05, 0.5], btype="band", fs=fs, output="sos")
        filtered    = signal.sosfilt(sos, spatial_avg)
        bandpass_rms = float(np.sqrt(np.mean(filtered ** 2)))
    except Exception:
        bandpass_rms = np.nan

    # 10. Bandpass variance — variance of the filtered signal
    #     More directly comparable to Hs (which scales with sqrt of variance)
    try:
        bandpass_variance = float(np.var(filtered))
    except Exception:
        bandpass_variance = np.nan

    return {
        "variance":             variance,
        "rms":                  rms,
        "dominant_period_s":    dominant_period,
        "wave_band_energy":     wave_band_energy,
        "shore_power":          shore_power,
        "phase_velocity_ms":    phase_velocity,
        "spatial_coherence":    spatial_coherence,
        "temporal_variability": temporal_variability,
        "bandpass_rms":         bandpass_rms,
        "bandpass_variance":    bandpass_variance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LABELS
# ─────────────────────────────────────────────────────────────────────────────

def label_condition(hs):
    if np.isnan(hs):
        return None
    if hs < HS_CALM_MAX:
        return "calm"
    elif hs >= HS_STORMY_MIN:
        return "stormy"
    else:
        return "swell"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def parse_timestamp(filepath):
    name   = os.path.basename(filepath)
    ts_str = name.replace("decimated_", "").replace(".pkl", "")
    dt     = datetime.strptime(ts_str, "%Y%m%d_%H%M")
    return dt.replace(tzinfo=timezone.utc)


def build_dataset(das_dir, wave_lookup):
    files = sorted(glob.glob(os.path.join(das_dir, "**", "decimated_*.pkl"), recursive=True))
    files = [f for f in files if "Copy" not in f and " - " not in f]
    print(f"\nFound {len(files)} DAS files")

    rows = []
    for fp in files:
        ts = parse_timestamp(fp)
        with open(fp, "rb") as f:
            d = pickle.load(f)
        feats = extract_features(d["xx"], d["tt"], d["data"])
        hs    = get_hs(wave_lookup, ts)
        label = label_condition(hs)
        rows.append({"filepath": fp, "timestamp": ts, "date": ts.date(),
                     "hs": hs, "label": label, **feats})
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')} | Hs={hs:.2f}m | {str(label):<6} | "
              f"var={feats['variance']:.0f}  rms={feats['rms']:.1f}")

    df = pd.DataFrame(rows).dropna(subset=["label"]).copy()
    print(f"\nLabelled samples: {len(df)}")
    print(df["label"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "variance", "rms", "dominant_period_s", "wave_band_energy",
    "shore_power", "phase_velocity_ms", "spatial_coherence", "temporal_variability", "bandpass_rms", "bandpass_variance"
]


def train_classifier(df):
    print("\n" + "="*55)
    print("CLASSIFIER")
    print("="*55)

    X      = df[FEATURE_COLS].values
    le     = LabelEncoder()
    y      = le.fit_transform(df["label"])
    groups = pd.factorize(df["date"].astype(str))[0]
    clf    = RandomForestClassifier(n_estimators=300, random_state=42)

    n_dates = len(np.unique(groups))
    if n_dates > 1:
        y_pred = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())
        print("Leave-one-date-out cross-validation results:")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y      = y_te
        print("Single-date 70/30 split results:")

    print(classification_report(y, y_pred, target_names=le.classes_))
    cm = pd.DataFrame(confusion_matrix(y, y_pred),
                      index=le.classes_, columns=le.classes_)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm)

    # Refit on all data for importance
    clf.fit(X, le.transform(df["label"]))
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(imp.round(3).to_string())

    fig, ax = plt.subplots(figsize=(9, 4))
    imp.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Feature importance — wave condition classifier")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("\nSaved: feature_importance.png")
    return clf, le


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — REGRESSOR
# ─────────────────────────────────────────────────────────────────────────────

def train_regressor(df):
    print("\n" + "="*55)
    print("REGRESSOR  (predict Hs from DAS features)")
    print("="*55)

    X      = df[FEATURE_COLS].values
    y      = df["hs"].values
    groups = pd.factorize(df["date"].astype(str))[0]
    reg    = RandomForestRegressor(n_estimators=300, random_state=42)

    if len(np.unique(groups)) > 1:
        y_pred = cross_val_predict(reg, X, y, groups=groups, cv=LeaveOneGroupOut())
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        y      = y_te

    mae = mean_absolute_error(y, y_pred)
    r2  = r2_score(y, y_pred)
    print(f"MAE = {mae:.3f} m    R2 = {r2:.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, alpha=0.7, color="steelblue", edgecolors="white", s=60)
    lo = min(y.min(), y_pred.min()) - 0.05
    hi = max(y.max(), y_pred.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("True Hs (m)")
    ax.set_ylabel("Predicted Hs (m)")
    ax.set_title(f"Hs regression   MAE={mae:.2f} m   R2={r2:.2f}")
    plt.tight_layout()
    plt.savefig("hs_regression.png", dpi=150)
    print("Saved: hs_regression.png")
    return reg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — VISUALISE calm vs stormy
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_comparison(df):
    labels_present = df["label"].unique()
    if len(labels_present) < 2:
        print("Skipping comparison plot, only one class in dataset")
        return

    fig, axes = plt.subplots(1, len(labels_present), figsize=(7*len(labels_present), 5))
    if len(labels_present) == 1:
        axes = [axes]

    for ax, lbl in zip(axes, sorted(labels_present)):
        row = df[df["label"] == lbl].iloc[len(df[df["label"]==lbl])//2]
        with open(row["filepath"], "rb") as f:
            d = pickle.load(f)
        xx = d["xx"] / 1000.0
        tt = d["tt"]
        data = d["data"]
        vmax = np.nanpercentile(np.abs(data), 98)
        ax.pcolormesh(xx, tt, data, cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, shading="auto")
        ax.axvline(ACTIVE_MIN_M/1000, color="yellow", lw=0.8, ls="--", label="active region")
        ax.axvline(ACTIVE_MAX_M/1000, color="yellow", lw=0.8, ls="--")
        ax.set_xlabel("Distance from interrogator (km)")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"{lbl.upper()}  |  {row['timestamp'].strftime('%Y-%m-%d %H:%M')} UTC  |  Hs={row['hs']:.2f} m")

    plt.suptitle("DAS space-time plot by wave condition", fontsize=13)
    plt.tight_layout()
    plt.savefig("das_class_comparison.png", dpi=150)
    print("Saved: das_class_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Loading wave model ===")
    wave_lookup = load_wave_model(WAVE_MODEL_PATH, WAVE_MODEL_DIST_KM)

    print("\n=== Building feature dataset ===")
    df = build_dataset(DAS_DIR, wave_lookup)
    df.to_csv("das_features.csv", index=False)
    print("Saved: das_features.csv")

    print("\n=== Plotting class comparison ===")
    plot_class_comparison(df)

    print("\n=== Training classifier ===")
    clf, le = train_classifier(df)

    print("\n=== Training regressor ===")
    reg = train_regressor(df)

    print("\n=== Done! Output files ===")
    for f in ["das_features.csv", "das_class_comparison.png",
              "feature_importance.png", "hs_regression.png"]:
        print(f"  {f}")
