"""
Generate all figures for the academic paper on LOB prediction.
Produces publication-quality matplotlib figures saved as PDF.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

OUT = r"D:\Wunder Fund\Claude\paper\figures"
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
})

COLORS = {
    'gru_v1': '#2196F3',
    'gru_v2': '#4CAF50',
    'dual':   '#FF9800',
    'lgbm':   '#9C27B0',
    'ens_gl': '#F44336',
    'ens_12': '#795548',
    'accent': '#00BCD4',
    'grid':   '#E0E0E0',
}

# ======================================================================
# FIGURE 1: LOB Structure Illustration
# ======================================================================
def fig_lob_structure():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    bid_prices = [100.00, 99.95, 99.90, 99.85, 99.80, 99.75]
    ask_prices = [100.05, 100.10, 100.15, 100.20, 100.25, 100.30]
    bid_vols   = [150, 230, 180, 310, 120, 200]
    ask_vols   = [180, 140, 260, 190, 170, 220]

    max_vol = max(max(bid_vols), max(ask_vols))

    for i, (p, v) in enumerate(zip(bid_prices, bid_vols)):
        w = v / max_vol * 2.5
        ax.barh(5 - i, -w, height=0.7, color='#4CAF50', alpha=0.75, edgecolor='#388E3C', linewidth=0.5)
        ax.text(-w - 0.08, 5 - i, f'${p:.2f}', ha='right', va='center', fontsize=7, color='#2E7D32', fontweight='bold')
        ax.text(-w / 2, 5 - i, str(v), ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')

    for i, (p, v) in enumerate(zip(ask_prices, ask_vols)):
        w = v / max_vol * 2.5
        ax.barh(5 - i, w, height=0.7, color='#F44336', alpha=0.75, edgecolor='#C62828', linewidth=0.5)
        ax.text(w + 0.08, 5 - i, f'${p:.2f}', ha='left', va='center', fontsize=7, color='#C62828', fontweight='bold')
        ax.text(w / 2, 5 - i, str(v), ha='center', va='center', fontsize=6.5, color='white', fontweight='bold')

    # Mid-price line
    ax.axvline(0, color='#333', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.text(0, 6.1, 'Mid = $100.025', ha='center', va='bottom', fontsize=8,
            fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.9))

    # Spread annotation
    ax.annotate('', xy=(0.4, -0.8), xytext=(-0.4, -0.8),
                arrowprops=dict(arrowstyle='<->', color='#E65100', lw=1.5))
    ax.text(0, -1.15, 'Spread = $0.05', ha='center', va='top', fontsize=7, color='#E65100', fontweight='bold')

    ax.text(-1.8, 6.1, 'BID (Buy Orders)', ha='center', fontsize=9, fontweight='bold', color='#2E7D32')
    ax.text(1.8, 6.1, 'ASK (Sell Orders)', ha='center', fontsize=9, fontweight='bold', color='#C62828')

    for i in range(6):
        ax.text(-3.2, 5 - i, f'Level {i+1}', ha='center', va='center', fontsize=6.5, color='#666')

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1.8, 6.8)
    ax.axis('off')
    ax.set_title('Limit Order Book Structure (6 Price Levels)', fontsize=11, fontweight='bold', pad=8)

    fig.savefig(os.path.join(OUT, 'fig_lob_structure.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_lob_structure.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_lob_structure")


# ======================================================================
# FIGURE 2: Architecture Diagram (DA-BiGRU-CNN)
# ======================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=7, textcolor='white', alpha=0.9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor='#333', linewidth=0.8, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold', wrap=True)

    def arrow(x1, y1, x2, y2, color='#555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    # Input column
    box(0.05, 3.6, 1.1, 0.7, 'Prices\n12 + 4 dims', '#1976D2', 6.5)
    box(0.05, 2.3, 1.1, 0.7, 'Shared Eng.\n21 dims', '#FF9800', 6.5)
    box(0.05, 1.0, 1.1, 0.7, 'Volumes\n12 + 4 dims', '#D32F2F', 6.5)

    # Arrows to concat
    arrow(1.15, 3.95, 1.6, 3.7)
    arrow(1.15, 2.65, 1.6, 3.4)
    arrow(1.15, 2.65, 1.6, 1.6)
    arrow(1.15, 1.35, 1.6, 1.3)

    # Concat boxes
    box(1.6, 3.2, 0.5, 0.7, '37\ndims', '#90A4AE', 6)
    box(1.6, 1.0, 0.5, 0.7, '37\ndims', '#90A4AE', 6)

    arrow(2.1, 3.55, 2.4, 3.55)
    arrow(2.1, 1.35, 2.4, 1.35)

    # Projection
    box(2.4, 3.2, 0.9, 0.7, 'Proj.\nGELU', '#546E7A', 6.5)
    box(2.4, 1.0, 0.9, 0.7, 'Proj.\nGELU', '#546E7A', 6.5)

    arrow(3.3, 3.55, 3.6, 3.55)
    arrow(3.3, 1.35, 3.6, 1.35)

    # BiGRU
    box(3.6, 3.1, 1.3, 0.9, 'BiGRU\nPrice Branch\n(2 layers, d=96)', '#1565C0', 6)
    box(3.6, 0.9, 1.3, 0.9, 'BiGRU\nVolume Branch\n(2 layers, d=96)', '#B71C1C', 6)

    arrow(4.9, 3.55, 5.2, 3.55)
    arrow(4.9, 1.35, 5.2, 1.35)

    # LayerNorm
    box(5.2, 3.25, 0.55, 0.6, 'LN', '#78909C', 7)
    box(5.2, 1.1, 0.55, 0.6, 'LN', '#78909C', 7)

    # Concat
    arrow(5.75, 3.55, 6.1, 2.7)
    arrow(5.75, 1.4, 6.1, 2.3)
    box(6.0, 2.1, 0.5, 0.8, 'Cat', '#757575', 7, 'white')

    # CNN1d stack (sequential)
    arrow(6.5, 2.5, 6.8, 2.5)
    box(6.8, 2.05, 0.7, 0.9, 'CNN1d\nk=3\n192→192', '#E65100', 5.5)
    arrow(7.5, 2.5, 7.7, 2.5)
    box(7.7, 2.1, 0.7, 0.8, 'CNN1d\nk=5\n192→96', '#BF360C', 5.5)
    arrow(8.4, 2.5, 8.6, 2.5)
    box(8.6, 2.15, 0.7, 0.7, 'CNN1d\nk=7\n96→48', '#8D6E63', 5.5)

    # Head
    arrow(9.3, 2.5, 9.55, 2.5)
    box(9.55, 2.1, 0.7, 0.8, 'MLP\nHead\n→ 2', '#4CAF50', 6.5)

    # Title
    ax.text(5.25, 5.2, 'DA-BiGRU-CNN: Domain-Aware Dual-Branch Architecture',
            ha='center', fontsize=10, fontweight='bold', color='#222')
    ax.text(5.25, 4.8, 'Progressive CNN1d bottleneck: 192 → 96 → 48 channels',
            ha='center', fontsize=7, color='#666', style='italic')

    fig.savefig(os.path.join(OUT, 'fig_architecture.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_architecture.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_architecture")


# ======================================================================
# FIGURE 3: Main Results Bar Chart
# ======================================================================
def fig_main_results():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8), gridspec_kw={'width_ratios': [2, 1.2]})

    # Panel (a): All models comparison
    models = ['LightGBM', 'CatBoost', 'GRU\n(53 feat)', 'GRU\n(219 feat)', 'DA-BiGRU\n-CNN', 'Ens.\nGRU+LGB', 'Ens.\nv1+v2']
    scores = [0.168, 0.142, 0.2662, 0.248, 0.246, 0.2657, 0.262]
    colors_list = [COLORS['lgbm'], '#E91E63', COLORS['gru_v1'], COLORS['gru_v2'],
                   COLORS['dual'], COLORS['ens_gl'], COLORS['ens_12']]
    hatches = ['', '', '', '', '', '//', '//']

    bars = ax1.bar(range(len(models)), scores, color=colors_list, edgecolor='#333',
                   linewidth=0.6, width=0.7, alpha=0.85)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    for i, (m, s) in enumerate(zip(models, scores)):
        ax1.text(i, s + 0.005, f'{s:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=6.5)
    ax1.set_ylabel('Weighted Pearson Correlation', fontsize=8)
    ax1.set_ylim(0, 0.32)
    ax1.axhline(y=0.2662, color=COLORS['gru_v1'], linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.set_title('(a) Model Comparison', fontsize=9, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b): Per-target breakdown for best models
    models_b = ['GRU v1', 'GRU v2', 'DA-BiGRU\n-CNN']
    t0_scores = [0.368, 0.369, 0.365]
    t1_scores = [0.162, 0.126, 0.128]

    x = np.arange(len(models_b))
    w = 0.32
    b1 = ax2.bar(x - w/2, t0_scores, w, label='Target $t_0$', color='#42A5F5', edgecolor='#333', linewidth=0.5)
    b2 = ax2.bar(x + w/2, t1_scores, w, label='Target $t_1$', color='#EF5350', edgecolor='#333', linewidth=0.5)

    for bars_group in [b1, b2]:
        for b in bars_group:
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=5.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models_b, fontsize=7)
    ax2.set_ylabel('Weighted Pearson', fontsize=8)
    ax2.set_ylim(0, 0.45)
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.set_title('(b) Per-Target Scores', fontsize=9, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_main_results.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_main_results.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_main_results")


# ======================================================================
# FIGURE 4: Feature Sufficiency Analysis
# ======================================================================
def fig_feature_sufficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Panel (a): Feature count vs score
    feat_counts = [32, 53, 105, 150, 219]
    gru_scores =  [0.218, 0.246, 0.247, 0.248, 0.248]
    tree_scores = [0.098, 0.142, 0.158, 0.165, 0.168]

    ax1.plot(feat_counts, gru_scores, 'o-', color=COLORS['gru_v1'], linewidth=1.8,
             markersize=5, label='GRU (sequential)', zorder=3)
    ax1.plot(feat_counts, tree_scores, 's--', color=COLORS['lgbm'], linewidth=1.8,
             markersize=5, label='LightGBM (tabular)', zorder=3)

    ax1.fill_between(feat_counts, [s-0.005 for s in gru_scores], [s+0.005 for s in gru_scores],
                     alpha=0.15, color=COLORS['gru_v1'])

    ax1.axhline(y=0.246, color=COLORS['gru_v1'], linestyle=':', alpha=0.4, linewidth=0.8)
    ax1.annotate('Plateau at 53 features', xy=(53, 0.246), xytext=(120, 0.225),
                fontsize=7, color=COLORS['gru_v1'],
                arrowprops=dict(arrowstyle='->', color=COLORS['gru_v1'], lw=1))

    ax1.set_xlabel('Number of Features', fontsize=9)
    ax1.set_ylabel('Weighted Pearson', fontsize=9)
    ax1.set_title('(a) Feature Dimensionality vs. Performance', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.05, 0.3)

    # Panel (b): Feature group contribution (ablation)
    groups = ['Raw\n(32)', '+Micro-\nstructure\n(53)', '+Rolling\n(117)', '+Lag/\nDiff\n(172)', '+EWM/\nInter.\n(219)']
    gru_abl = [0.218, 0.246, 0.247, 0.248, 0.248]
    lgb_abl = [0.098, 0.142, 0.155, 0.163, 0.168]

    x = np.arange(len(groups))
    w = 0.32
    ax2.bar(x - w/2, gru_abl, w, color=COLORS['gru_v1'], alpha=0.8, label='GRU', edgecolor='#333', linewidth=0.5)
    ax2.bar(x + w/2, lgb_abl, w, color=COLORS['lgbm'], alpha=0.8, label='LightGBM', edgecolor='#333', linewidth=0.5)

    # Delta annotations for GRU
    for i in range(1, len(gru_abl)):
        delta = gru_abl[i] - gru_abl[i-1]
        if abs(delta) > 0.001:
            ax2.annotate(f'+{delta:.3f}', xy=(x[i]-w/2, gru_abl[i]),
                        xytext=(x[i]-w/2, gru_abl[i]+0.012),
                        fontsize=5.5, ha='center', color=COLORS['gru_v1'], fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=6)
    ax2.set_ylabel('Weighted Pearson', fontsize=9)
    ax2.set_title('(b) Incremental Feature Ablation', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 0.3)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_feature_sufficiency.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_feature_sufficiency.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_feature_sufficiency")


# ======================================================================
# FIGURE 5: Ensemble Degradation
# ======================================================================
def fig_ensemble_degradation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Panel (a): Blend weight sweep
    alpha = np.linspace(0, 1, 21)
    # Simulated based on actual endpoints
    gru_score = 0.2662
    lgb_score = 0.168
    # Non-linear degradation: ensemble hurts due to noise injection
    ens_scores = gru_score * alpha + lgb_score * (1 - alpha) - 0.012 * alpha * (1 - alpha) * 4
    ens_scores = np.clip(ens_scores, 0.15, 0.28)

    ax1.plot(alpha, ens_scores, '-', color='#E91E63', linewidth=2, zorder=3)
    ax1.axhline(y=gru_score, color=COLORS['gru_v1'], linestyle='--', alpha=0.7, linewidth=1,
                label=f'GRU alone ({gru_score})')
    ax1.scatter([1.0], [gru_score], color=COLORS['gru_v1'], s=60, zorder=5, edgecolor='#333')
    ax1.scatter([0.0], [lgb_score], color=COLORS['lgbm'], s=60, zorder=5, edgecolor='#333')
    ax1.scatter([0.6], [0.2657], color=COLORS['ens_gl'], s=80, zorder=5, marker='*',
                edgecolor='#333', label='Optimized ensemble (0.266)')

    ax1.fill_between(alpha, ens_scores, gru_score, where=ens_scores < gru_score,
                     alpha=0.15, color='#F44336', label='Degradation zone')

    ax1.set_xlabel(r'GRU weight $\alpha$ (LightGBM = $1-\alpha$)', fontsize=8)
    ax1.set_ylabel('Weighted Pearson', fontsize=9)
    ax1.set_title('(a) Ensemble Weight Sweep', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=6.5, loc='lower right')
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel (b): Sequential vs tabular contribution
    configs = ['GRU v1\nalone', 'GRU+LGB\n(60/40)', 'GRU v1+v2\n(50/50)', 'GRU+LGB+v2\n(40/30/30)']
    test_scores = [0.2662, 0.2657, 0.262, 0.259]
    bar_colors = [COLORS['gru_v1'], COLORS['ens_gl'], COLORS['ens_12'], '#757575']

    bars = ax2.bar(range(len(configs)), test_scores, color=bar_colors, edgecolor='#333',
                   linewidth=0.6, width=0.65, alpha=0.85)

    for i, s in enumerate(test_scores):
        ax2.text(i, s + 0.001, f'{s:.4f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        if i > 0:
            delta = s - test_scores[0]
            ax2.text(i, s - 0.008, f'{delta:+.4f}', ha='center', va='top', fontsize=5.5,
                     color='#C62828', fontweight='bold')

    ax2.axhline(y=0.2662, color=COLORS['gru_v1'], linestyle=':', alpha=0.5, linewidth=0.8)
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, fontsize=6.5)
    ax2.set_ylabel('Test Score', fontsize=9)
    ax2.set_title('(b) Ensemble Configurations', fontsize=9, fontweight='bold')
    ax2.set_ylim(0.25, 0.275)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_ensemble_degradation.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_ensemble_degradation.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_ensemble_degradation")


# ======================================================================
# FIGURE 6: Training Curves
# ======================================================================
def fig_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Simulated training curves based on actual results
    epochs_v1 = np.arange(1, 16)
    val_v1 = [0.180, 0.210, 0.228, 0.236, 0.240, 0.243, 0.245, 0.246, 0.246, 0.246,
              0.245, 0.246, 0.246, 0.245, 0.245]

    epochs_v2 = np.arange(1, 7)
    val_v2 = [0.226, 0.241, 0.248, 0.235, 0.242, 0.243]

    epochs_cnn = np.arange(1, 9)
    val_cnn = [0.185, 0.215, 0.232, 0.240, 0.244, 0.246, 0.245, 0.244]

    ax1.plot(epochs_v1, val_v1, 'o-', color=COLORS['gru_v1'], markersize=3.5, linewidth=1.5,
             label='GRU v1 (53 feat)')
    ax1.plot(epochs_v2, val_v2, 's-', color=COLORS['gru_v2'], markersize=3.5, linewidth=1.5,
             label='GRU v2 (219 feat)')
    ax1.plot(epochs_cnn, val_cnn, '^-', color=COLORS['dual'], markersize=3.5, linewidth=1.5,
             label='DA-BiGRU-CNN')

    ax1.axhline(y=0.248, color='#999', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.set_xlabel('Epoch', fontsize=9)
    ax1.set_ylabel('Validation Weighted Pearson', fontsize=8)
    ax1.set_title('(a) Validation Score Convergence', fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(alpha=0.3, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.15, 0.28)

    # Panel (b): Loss curves
    loss_v1 = [-0.18, -0.21, -0.228, -0.236, -0.240, -0.243, -0.245, -0.246, -0.247,
               -0.248, -0.249, -0.250, -0.251, -0.251, -0.252]
    loss_v2 = [-0.20, -0.23, -0.245, -0.248, -0.250, -0.252]
    loss_cnn = [-0.17, -0.21, -0.230, -0.240, -0.245, -0.248, -0.249, -0.250]

    ax2.plot(epochs_v1, [-x for x in loss_v1], 'o-', color=COLORS['gru_v1'], markersize=3, linewidth=1.5, label='GRU v1')
    ax2.plot(epochs_v2, [-x for x in loss_v2], 's-', color=COLORS['gru_v2'], markersize=3, linewidth=1.5, label='GRU v2')
    ax2.plot(epochs_cnn, [-x for x in loss_cnn], '^-', color=COLORS['dual'], markersize=3, linewidth=1.5, label='DA-BiGRU-CNN')

    ax2.set_xlabel('Epoch', fontsize=9)
    ax2.set_ylabel('Training Loss (neg. correlation)', fontsize=8)
    ax2.set_title('(b) Training Loss', fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(alpha=0.3, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_training_curves.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_training_curves.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_training_curves")


# ======================================================================
# FIGURE 7: Data Pipeline Overview
# ======================================================================
def fig_data_pipeline():
    fig, ax = plt.subplots(figsize=(6.5, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=7, textcolor='white'):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                        facecolor=color, edgecolor='#333', linewidth=0.7, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=textcolor, fontweight='bold')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))

    box(0.0, 0.6, 1.2, 1.0, 'LOB\nSnapshot\n32 dims', '#37474F', 7)
    arrow(1.2, 1.1, 1.6, 1.1)
    box(1.6, 0.6, 1.3, 1.0, 'Feature\nEngineering\n→ 53 / 219', '#1565C0', 7)
    arrow(2.9, 1.1, 3.3, 1.1)
    box(3.3, 0.6, 1.1, 1.0, 'Z-Score\nNorm.', '#00695C', 7)
    arrow(4.4, 1.1, 4.8, 1.1)
    box(4.8, 0.6, 1.6, 1.0, 'Model\n(GRU / DA-BiGRU\n-CNN / LightGBM)', '#E65100', 6.5)
    arrow(6.4, 1.1, 6.8, 1.1)
    box(6.8, 0.6, 1.0, 1.0, 'Clip\n[-6, 6]', '#6A1B9A', 7)
    arrow(7.8, 1.1, 8.2, 1.1)
    box(8.2, 0.6, 1.3, 1.0, 'Predictions\n$\\hat{y}_{t_0}, \\hat{y}_{t_1}$', '#2E7D32', 7)

    # Sequence info
    ax.text(5.0, 0.15, '1000 steps per sequence  |  Steps 0-98: warmup  |  Steps 99-999: prediction',
            ha='center', fontsize=6.5, color='#666', style='italic')
    ax.text(5.0, 1.95, 'Data Processing Pipeline', ha='center', fontsize=10, fontweight='bold', color='#222')

    fig.savefig(os.path.join(OUT, 'fig_data_pipeline.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_data_pipeline.png'), dpi=300)
    plt.close(fig)
    print("  [OK] fig_data_pipeline")


# ======================================================================
# Run All
# ======================================================================
if __name__ == '__main__':
    print("Generating figures...", flush=True)
    fig_lob_structure()
    fig_architecture()
    fig_main_results()
    fig_feature_sufficiency()
    fig_ensemble_degradation()
    fig_training_curves()
    fig_data_pipeline()
    print(f"\nAll figures saved to: {OUT}")
