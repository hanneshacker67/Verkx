
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────

DATE            = "20241008"   # <-- BREYTTU í "20241012" fyrir 12. október
DAS_DIR         = "/mnt/c/Users/Hanne/Documents/verkx/datafiles"
WAVE_MODEL_PATH = "/mnt/c/Users/Hanne/Documents/verkx/storm_oct_2024/wave_data_100km_oct2024.pkl"
OUTPUT_DIR      = "/mnt/c/Users/Hanne/Documents/verkx/pipeline_output"
WAVE_DIST_KM    = 9.2
ACTIVE_MIN_M    = 10_000
ACTIVE_MAX_M    = 11_500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load WAM ─────────────────────────────────────────────────────────────────

print("Hleð WAM gögnum...")
with open(WAVE_MODEL_PATH, "rb") as f:
    wam = pickle.load(f)

dist        = np.array(wam["dist"])
hs_grid     = np.array(wam["wave_hs"])
times       = pd.to_datetime(wam["time"].values)
dist_idx    = np.argmin(np.abs(dist - WAVE_DIST_KM))
hs_at_cable = hs_grid[:, dist_idx]

wam_df = pd.DataFrame({
    "hour_utc": times,
    "Hs_m":     hs_at_cable
}).dropna()

date_dt  = pd.to_datetime(DATE, format="%Y%m%d")
wam_day  = wam_df[wam_df["hour_utc"].dt.date == date_dt.date()].copy()
print(f"  WAM gildi fyrir {DATE}: {len(wam_day)} klukkustundir")

# ── Group Das files────────────────────────────────

print(f"Finn DAS skrár fyrir {DATE}...")
day_dir   = os.path.join(DAS_DIR, DATE)
das_files = sorted([f for f in os.listdir(day_dir)
                    if f.startswith("decimated_") and f.endswith(".pkl")])
print(f"  Fann {len(das_files)} skrár")


hour_to_files = defaultdict(list)
for fname in das_files:
    parts = fname.replace("decimated_", "").replace(".pkl", "").split("_")
    dt    = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M")
    hour  = dt.replace(minute=0, second=0)
    hour_to_files[hour].append(os.path.join(day_dir, fname))

# ── Calculate RMS ───────────────────────────────

print("Reikna hourly RMS")
hourly_records = []

for hour, files in sorted(hour_to_files.items()):
    all_samples = []
    for fpath in sorted(files):
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        mask   = (d["xx"] >= ACTIVE_MIN_M) & (d["xx"] <= ACTIVE_MAX_M)
        active = d["data"][:, mask]
        all_samples.append(active.flatten())

    all_samples = np.concatenate(all_samples)
    rms = float(np.sqrt(np.mean(all_samples ** 2)))
    hourly_records.append({"hour_utc": hour, "rms": rms})

das_hourly = pd.DataFrame(hourly_records)
print(f"  {len(das_hourly)} klukkustundir reiknað")

# ── Combine DAS and WAM ────────────────────────────────────────────────────────

wam_day["hour_utc"] = pd.to_datetime(wam_day["hour_utc"])
das_hourly["hour_utc"] = pd.to_datetime(das_hourly["hour_utc"])
matched = pd.merge(das_hourly, wam_day, on="hour_utc", how="inner")
print(f"  {len(matched)} samsvöruð pör")

# ── Fig ──────────────────────────────────────────────────────────────────────

date_label = date_dt.strftime("%d. october %Y")
fig, axes  = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f"DAS RMS vs WAM Hs — {date_label}", 
             fontsize=14, fontweight="bold")

ax  = axes[0]
ax2 = ax.twinx()

ax.plot(matched["hour_utc"], matched["rms"],
        "o-", color="steelblue", lw=2, ms=6,
        label=f"DAS Hourly RMS )")
ax2.plot(matched["hour_utc"], matched["Hs_m"],
         "o-", color="tomato", lw=2, ms=6,
         label=f"WAM Hs við {dist[dist_idx]:.1f} km")

ax.set_ylabel("DAS RMS (arbitrary units)", color="steelblue", fontsize=11)
ax.tick_params(axis="y", labelcolor="steelblue")
ax.set_xlabel("Hours")
ax2.set_ylabel("WAM Significant Wave Height (m)", color="tomato", fontsize=11)
ax2.tick_params(axis="y", labelcolor="tomato")

ax.grid(alpha=0.3)
fig.autofmt_xdate()

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper left")



plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, f"rms_vs_hs_{DATE}.png")
plt.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"\nVistað: {outpath}")
