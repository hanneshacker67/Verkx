import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import find_peaks

# ── Load all three files ──────────────────────────────────────────────────────

files = [
    r'C:\Users\Hanne\Documents\verkx\datafiles\decimated_20241004_1200.pkl',
    r'C:\Users\Hanne\Documents\verkx\datafiles\decimated_20241004_1205.pkl',
    r'C:\Users\Hanne\Documents\verkx\datafiles\decimated_20241004_1210.pkl',
]

labels = ['12:00', '12:05', '12:10']
datasets = []
for f in files:
    with open(f, 'rb') as fh:
        datasets.append(pickle.load(fh))

d0 = datasets[0]
xx = d0['xx']          # distance along cable (m)
tt = d0['tt']          # time within 5-min window (s)
data = d0['data']      # shape: (1200 time steps, 587 channels)

dx = xx[1] - xx[0]    # ~10.2 m spacing
dt = tt[1] - tt[0]    # 0.25 s  →  4 Hz

norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
cmap = 'RdBu_r'

STYLE = dict(facecolor='#0A1628')
plt.rcParams.update({
    'figure.facecolor': '#0A1628',
    'axes.facecolor':   '#0D1F3C',
    'axes.edgecolor':   '#334155',
    'axes.labelcolor':  'white',
    'xtick.color':      'white',
    'ytick.color':      'white',
    'text.color':       'white',
    'grid.color':       '#334155',
    'grid.linewidth':   0.5,
})

TEAL  = '#14B8A6'
MINT  = '#5EEAD4'
GOLD  = '#F59E0B'
RED   = '#F87171'
WHITE = 'white'

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — The full space-time picture  (what IS the data?)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('#0A1628')
ax.set_facecolor('#0D1F3C')

im = ax.pcolormesh(xx/1000, tt, data, cmap=cmap, norm=norm, shading='auto', rasterized=True)
cb = plt.colorbar(im, ax=ax, label='Strain rate (με/s)', pad=0.02)
cb.ax.yaxis.label.set_color('white')
cb.ax.tick_params(colors='white')

# Annotate: horizontal slice we'll show in Fig 2
t_slice = 100  # seconds
ax.axhline(t_slice, color=GOLD, lw=1.5, ls='--', label=f't = {t_slice} s (used in Fig 2)')

# Annotate: vertical slice we'll show in Fig 3
x_chan = 10500  # metres — near shore, active signal
ax.axvline(x_chan/1000, color=MINT, lw=1.5, ls='--', label=f'x = {x_chan} m (used in Fig 3)')

ax.set_xlabel('Distance along cable (km)', fontsize=12)
ax.set_ylabel('Time (s)', fontsize=12)
ax.set_title('Figure 1 — Full space–time view of the DAS data (12:00 UTC)\n'
             'Each pixel = one sensor at one moment. Diagonal stripes are individual waves.',
             fontsize=11, pad=10)
ax.legend(fontsize=9, framealpha=0.4)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\Hanne\Documents\verkx\fig1_spacetime.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Snapshot in space (one horizontal slice)
#            Shows wavelength shortening toward shore (Siggi's observation)
# ═══════════════════════════════════════════════════════════════════════════════
t_idx = np.argmin(np.abs(tt - t_slice))
spatial_slice = data[t_idx, :]          # one row: signal at all positions at t=100s

fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'hspace': 0.5})
fig.patch.set_facecolor('#0A1628')

# ── Top panel: the slice itself ───────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#0D1F3C')
ax.plot(xx/1000, spatial_slice, color=TEAL, lw=0.9, alpha=0.9)
ax.axhline(0, color='white', lw=0.5, alpha=0.4)
ax.set_ylabel('Strain rate (με/s)', fontsize=11)
ax.set_xlabel('Distance along cable (km)', fontsize=11)
ax.set_title(f'Figure 2 (top) — Spatial snapshot at t = {t_slice} s\n'
             'This is what the whole cable "feels" at one instant.', fontsize=10, pad=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(xx.min()/1000, xx.max()/1000)

# Mark shore-side vs offshore
ax.axvspan(xx.min()/1000, 10.8, alpha=0.07, color=RED,  label='Near shore (shallow)')
ax.axvspan(10.8,  xx.max()/1000, alpha=0.07, color='blue', label='Offshore (deeper)')
ax.legend(fontsize=9, framealpha=0.4)

# ── Bottom panel: zoom into shore region to show bunching ────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0D1F3C')

# Two zoom windows to compare wavelength
zoom1_mask = (xx >= 10000) & (xx <= 10600)   # near shore
zoom2_mask = (xx >= 11500) & (xx <= 12500)   # offshore

ax2.plot(xx[zoom1_mask]/1000, spatial_slice[zoom1_mask], color=RED,  lw=1.5, label='Near shore (~10–10.6 km)')
ax2.plot(xx[zoom2_mask]/1000, spatial_slice[zoom2_mask], color=MINT, lw=1.5, label='Offshore (~11.5–12.5 km)')
ax2.axhline(0, color='white', lw=0.5, alpha=0.4)

# Estimate peak-to-peak spacing for both regions and annotate
for mask, color, region_name in [(zoom1_mask, RED, 'near shore'), (zoom2_mask, MINT, 'offshore')]:
    seg = spatial_slice[mask]
    seg_x = xx[mask]
    peaks, _ = find_peaks(seg, height=5, distance=3)
    if len(peaks) >= 2:
        spacings = np.diff(seg_x[peaks])
        mean_lam = spacings.mean()
        for pk in peaks[:4]:
            ax2.axvline(seg_x[pk]/1000, color=color, lw=0.8, ls=':', alpha=0.6)

ax2.set_xlabel('Distance along cable (km)', fontsize=11)
ax2.set_ylabel('Strain rate (με/s)', fontsize=11)
ax2.set_title('Figure 2 (bottom) — Zoom: comparing wavelength near shore vs offshore\n'
              'Stripes are closer together near shore → waves "bunching up" in shallow water (Siggi\'s observation)',
              fontsize=10, pad=8)
ax2.legend(fontsize=9, framealpha=0.4)
ax2.grid(True, alpha=0.3)

plt.savefig(r'C:\Users\Hanne\Documents\verkx\'fig2_spatial_snapshot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Time series at a single channel
#            What does one "sensor" see over 5 minutes?
# ═══════════════════════════════════════════════════════════════════════════════
x_idx = np.argmin(np.abs(xx - x_chan))
time_series = data[:, x_idx]           # one column: signal at x=10500m over time

fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'hspace': 0.5})
fig.patch.set_facecolor('#0A1628')

# ── Full 5-minute time series ─────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#0D1F3C')
ax.plot(tt, time_series, color=TEAL, lw=0.8, alpha=0.9)
ax.axhline(0, color='white', lw=0.5, alpha=0.4)
ax.set_xlabel('Time (s)', fontsize=11)
ax.set_ylabel('Strain rate (με/s)', fontsize=11)
ax.set_title(f'Figure 3 (top) — Time series at x = {x_chan/1000:.1f} km (one sensor, full 5 min)\n'
             'This is what a single point on the cable records. You can see individual wave crests.',
             fontsize=10, pad=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, tt.max())

# Find peaks (wave crests) and mark them
peaks, props = find_peaks(time_series, height=30, distance=8)
if len(peaks) > 0:
    ax.plot(tt[peaks], time_series[peaks], 'o', color=GOLD, ms=5, label=f'{len(peaks)} wave crests detected', zorder=5)
    ax.legend(fontsize=9, framealpha=0.4)

    # Estimate wave period
    if len(peaks) >= 2:
        periods = np.diff(tt[peaks])
        mean_T = periods.mean()
        ax.text(0.98, 0.95, f'Mean wave period ≈ {mean_T:.1f} s', transform=ax.transAxes,
                ha='right', va='top', fontsize=10, color=GOLD,
                bbox=dict(boxstyle='round', facecolor='#0A1628', alpha=0.7))

# ── Zoom into 30 seconds to see individual waves clearly ─────────────────────
ax2 = axes[1]
ax2.set_facecolor('#0D1F3C')
t_zoom = (tt >= 50) & (tt <= 80)
ax2.plot(tt[t_zoom], time_series[t_zoom], color=TEAL, lw=1.5)
ax2.axhline(0, color='white', lw=0.5, alpha=0.4)

# Mark crests in zoom window
zoom_peaks = peaks[(tt[peaks] >= 50) & (tt[peaks] <= 80)]
if len(zoom_peaks) > 0:
    ax2.plot(tt[zoom_peaks], time_series[zoom_peaks], 'o', color=GOLD, ms=8, zorder=5)
    if len(zoom_peaks) >= 2:
        for i in range(len(zoom_peaks)-1):
            t1, t2 = tt[zoom_peaks[i]], tt[zoom_peaks[i+1]]
            ax2.annotate('', xy=(t2, time_series[zoom_peaks[i+1]]+15),
                        xytext=(t1, time_series[zoom_peaks[i]]+15),
                        arrowprops=dict(arrowstyle='<->', color=GOLD, lw=1.5))
            ax2.text((t1+t2)/2, time_series[zoom_peaks[i]]+22, f'{t2-t1:.1f} s',
                    ha='center', color=GOLD, fontsize=9)

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Strain rate (με/s)', fontsize=11)
ax2.set_title('Figure 3 (bottom) — Zoom: 30-second window showing individual wave crests\n'
              'Each gold dot is a wave crest. The spacing in time = wave period.',
              fontsize=10, pad=8)
ax2.grid(True, alpha=0.3)

plt.savefig('fig3_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Three files side by side (how does the wave field evolve?)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
fig.patch.set_facecolor('#0A1628')
fig.suptitle('Figure 4 — How the wave field changes over 15 minutes (12:00 → 12:10 UTC)\n'
             'Same cable, three consecutive 5-minute windows. The wave pattern is broadly stable — this is swell.',
             fontsize=11, y=1.01)

for i, (ds, label, ax) in enumerate(zip(datasets, labels, axes)):
    ax.set_facecolor('#0D1F3C')
    im = ax.pcolormesh(ds['xx']/1000, ds['tt'], ds['data'],
                       cmap=cmap, norm=norm, shading='auto', rasterized=True)
    ax.set_title(f'12:{label[3:]} UTC\nstd = {ds["data"].std():.1f} με/s', fontsize=10)
    ax.set_xlabel('Distance (km)', fontsize=10)
    if i == 0:
        ax.set_ylabel('Time (s)', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.colorbar(im, ax=axes[-1], label='Strain rate (με/s)').ax.tick_params(colors='white')
plt.tight_layout()
plt.savefig(r'C:\Users\Hanne\Documents\verkx\'fig4_three_files.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — What the data CANNOT tell us (limits)
#            Show the offshore dead zone + frequency content
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor('#0A1628')
fig.suptitle('Figure 5 — What the data can and cannot tell us', fontsize=12, y=1.01)

# Left: highlight the active vs quiet cable regions
ax = axes[0]
ax.set_facecolor('#0D1F3C')
# Mean absolute amplitude along cable
mean_amp = np.abs(data).mean(axis=0)
ax.fill_between(xx/1000, mean_amp, alpha=0.7, color=TEAL)
ax.axvline(11.5, color=RED, lw=2, ls='--', label='Signal drops off ~11.5 km')
ax.set_xlabel('Distance along cable (km)', fontsize=11)
ax.set_ylabel('Mean |strain rate| (με/s)', fontsize=11)
ax.set_title('Active cable region\n(mean amplitude along cable)', fontsize=10)
ax.legend(fontsize=9, framealpha=0.4)
ax.grid(True, alpha=0.3)
# Shade quiet zone
ax.axvspan(11.5, xx.max()/1000, alpha=0.15, color=RED, label='Low signal zone')
ax.text(13, mean_amp.max()*0.6, 'Low signal\n(deep water / no wave\ninteraction with cable)',
        ha='center', color=RED, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#0A1628', alpha=0.7))

# Right: frequency spectrum at the active channel
ax2 = axes[1]
ax2.set_facecolor('#0D1F3C')
from numpy.fft import rfft, rfftfreq
ts = data[:, x_idx]
freqs = rfftfreq(len(ts), d=dt)
power = np.abs(rfft(ts))**2
# Only show 0–0.5 Hz (wave frequencies)
mask = freqs <= 0.5
ax2.semilogy(freqs[mask], power[mask], color=TEAL, lw=1.2)
# Mark dominant peak
peak_f_idx = np.argmax(power[mask])
peak_f = freqs[mask][peak_f_idx]
peak_T = 1/peak_f if peak_f > 0 else 0
ax2.axvline(peak_f, color=GOLD, lw=1.5, ls='--',
            label=f'Dominant freq = {peak_f:.3f} Hz\n(Period ≈ {peak_T:.1f} s)')
ax2.set_xlabel('Frequency (Hz)', fontsize=11)
ax2.set_ylabel('Power spectral density', fontsize=11)
ax2.set_title(f'Frequency content at x = {x_chan/1000:.1f} km\n'
              'Tells us dominant wave period but NOT wave height directly', fontsize=10)
ax2.legend(fontsize=9, framealpha=0.4)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\Hanne\Documents\verkx\fig5_limits.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 saved")

print("\n✅ All 5 figures saved.")