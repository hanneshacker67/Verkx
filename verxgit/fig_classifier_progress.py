
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = '/mnt/c/Users/Hanne/Documents/verkx/fig_classifier_progress.png'

plt.rcParams.update({
    'figure.facecolor': '#0A1628', 'axes.facecolor': '#0D1F3C',
    'axes.edgecolor': '#334155', 'axes.labelcolor': 'white',
    'xtick.color': 'white', 'ytick.color': 'white',
    'text.color': 'white', 'grid.color': '#334155', 'grid.linewidth': 0.5,
})
TEAL = '#14B8A6'; GOLD = '#F59E0B'; GREY = '#64748B'; RED = '#F87171'; WHITE = 'white'

# ── Stage 1: majority-class baseline 
# ── Stage 2: three models, arbitrary thresholds 
#    LR 60.0%, RF 67.2%, ET 68.5%  (classification_results.png)
# ── Stage 3: k-means thresholds, top 20 features — RF 90.0%  
# ── Stage 4: feature selection 76→15, DAS only — RF 92.0%  (v3 final) 

fig, ax = plt.subplots(figsize=(14, 6.5))
fig.patch.set_facecolor('#0A1628')
ax.set_facecolor('#0D1F3C')


x_base   = 0
x_lr, x_rf, x_et = 1.1, 1.5, 1.9
x_kmeans = 3.0
x_final  = 4.2

bar_w = 0.32
bar_w_single = 0.55

# Baseline
ax.bar(x_base, 25.7, color=GREY, width=bar_w_single, alpha=0.9, zorder=3)
ax.text(x_base, 25.7 + 1.2, '25.7%', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=WHITE)

# Three models (arbitrary thresholds)
cols = ['#4B6FA8', '#3A7D6E', '#2C6B8A']
accs_mid = [60.0, 67.2, 68.5]
labels_mid = ['Logistic\nRegression', 'Random\nForest', 'Extra\nTrees']
for xi, acc, col, lbl in zip([x_lr, x_rf, x_et], accs_mid, cols, labels_mid):
    ax.bar(xi, acc, color=col, width=bar_w, alpha=0.92, zorder=3)
    ax.text(xi, acc + 1.0, f'{acc:.1f}%', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color=WHITE)

# K-means thresholds
ax.bar(x_kmeans, 90.0, color='#1C7293', width=bar_w_single, alpha=0.92, zorder=3)
ax.text(x_kmeans, 90.0 + 1.2, '90.0%', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color=WHITE)

# Final model
ax.bar(x_final, 92.0, color=TEAL, width=bar_w_single, alpha=0.95, zorder=3)
ax.text(x_final, 92.0 + 1.2, '92.0%', ha='center', va='bottom',
        fontsize=13, fontweight='bold', color=WHITE)

# ── Reference line ─────────────────────────────────────────────────────────────
ax.axhline(92.0, color=TEAL, lw=1.0, ls='--', alpha=0.3, zorder=2)

# ── Arrows between stages ──────────────────────────────────────────────────────
# baseline → ET (best of the three)
ax.annotate('', xy=(x_lr - 0.16, 60.0 + 0.5),
            xytext=(x_base + 0.28, 25.7 + 0.5),
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))
ax.text((x_base + x_lr) / 2, 52, '+34–43pp', ha='center', va='bottom',
        fontsize=9, color=GOLD, fontweight='bold')

# ET → k-means
ax.annotate('', xy=(x_kmeans - 0.28, 90.0 + 0.5),
            xytext=(x_et + 0.16, 68.5 + 0.5),
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))
ax.text((x_et + x_kmeans) / 2 + 0.1, 87, '+21.5pp', ha='center', va='bottom',
        fontsize=9, color=GOLD, fontweight='bold')

# k-means → final
ax.annotate('', xy=(x_final - 0.28, 92.0 + 0.5),
            xytext=(x_kmeans + 0.28, 90.0 + 0.5),
            arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5))
ax.text((x_kmeans + x_final) / 2, 97, '+2.0pp', ha='center', va='bottom',
        fontsize=9, color=GOLD, fontweight='bold')

# ── X-axis labels ──────────────────────────────────────────────────────────────
ax.set_xticks([x_base, (x_lr + x_et)/2, x_kmeans, x_final])
ax.set_xticklabels([
    'Majority-class\nbaseline',
    'Arbitrary thresholds\n(3 models tested)',
    '+ K-means\nthresholds',
    'Final model\n(feature selection)',
], fontsize=10.5)

# ── Detail annotations below bars ─────────────────────────────────────────────
ax.text(x_base, -7, 'Always predict "swell"', ha='center', va='top',
        fontsize=8, color='#94A3B8')
ax.text((x_lr + x_et)/2, -7, 'LR  60.0%   RF  67.2%   ET  68.5%',
        ha='center', va='top', fontsize=8, color='#94A3B8')
ax.text(x_kmeans, -7, 'Natural cluster boundaries\nTop 20 DAS features',
        ha='center', va='top', fontsize=8, color='#94A3B8', linespacing=1.4)
ax.text(x_final, -7, 'Random Forest, top 15\nDAS features only',
        ha='center', va='top', fontsize=8, color='#94A3B8', linespacing=1.4)

# ── Legend for the three middle bars ──────────────────────────────────────────
patches = [
    mpatches.Patch(color='#4B6FA8', label='Logistic Regression  60.0%'),
    mpatches.Patch(color='#3A7D6E', label='Random Forest  67.2%'),
    mpatches.Patch(color='#2C6B8A', label='Extra Trees  68.5%'),
]
ax.legend(handles=patches, fontsize=8.5, loc='upper left',
          facecolor='#0A1628', edgecolor='#334155', framealpha=0.8)

ax.set_ylabel('Classification accuracy (%)', fontsize=12)
ax.set_ylim(0, 108)
ax.set_xlim(-0.5, x_final + 0.55)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
ax.grid(axis='y', alpha=0.22, zorder=1)
ax.set_axisbelow(True)
ax.tick_params(axis='x', pad=48)
ax.set_title('Classifier development',
             fontsize=12, pad=14, color=WHITE)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}")
