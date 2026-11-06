"""Reflow frozen scene and spectrum arrays into a compact horizontal figure."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
source = root / 'analysis_v7_latex/figure_arrays.npz'
with np.load(source) as data:
    arrays = {k: data[k] for k in data.files}
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.titlesize': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.1), layout='constrained')
rx = np.stack([500*np.cos(np.arange(4)*np.pi/2), 500*np.sin(np.arange(4)*np.pi/2)], axis=1).astype(np.float32).astype(float)
ax = axes[0]
ax.scatter(rx[:, 0], rx[:, 1], marker='^', s=55, c='#247BA0')
for i, p in enumerate(rx):
    ax.annotate(f'站{i+1}', p, xytext=(5, 3), textcoords='offset points', fontsize=9)
for p, label, color, marker in zip(arrays['scene_positions_m'], ['强源', '弱源'], ['#D77D32', '#348753'], ['*', 'D']):
    ax.scatter(*p, marker=marker, s=85, c=color, edgecolors='black')
    ax.annotate(label, p, xytext=(6, -12), textcoords='offset points', fontsize=9)
ax.set(xlim=(-650, 700), ylim=(-650, 650), aspect='equal', title='（a）四站双源场景', xlabel='x / m', ylabel='y / m')
for i, overlap in enumerate([.5, 1.]):
    ax = axes[i+1]
    for density, label, color, style in zip(arrays[f'scene_psd_{overlap}'], ['强源 6 dBm', '弱源 0 dBm'], ['#D77D32', '#348753'], ['-', '--']):
        ax.plot(arrays['scene_frequency_mhz'], 10*np.log10(np.maximum(density, 1e-7)), lw=1.5, color=color, ls=style, label=label)
    width = 12.5
    shift = width*(1-overlap)/(1+overlap)
    for w, y, label in [(width-shift, -43, '交集'), (width+shift, -56, '并集')]:
        ax.annotate('', xy=(w/2, y), xytext=(-w/2, y), arrowprops={'arrowstyle': '|-|', 'lw': 1.1, 'color': '#334E68'})
        ax.text(0, y+1.6, f'{label} {w:.3f} MHz', ha='center', fontsize=8, bbox={'fc': 'white', 'ec': 'none', 'pad': .1})
    ax.set(xlim=(-11, 11), ylim=(-63, 7), title=f'（{"b" if i == 0 else "c"}）理想频谱  IoU = {overlap:g}', xlabel='相对频率 / MHz', ylabel='功率谱密度 / (dBm/MHz)')
    ax.legend(loc='upper right', fontsize=7)
for ax in axes:
    ax.grid(alpha=.2)
out = root / 'layout_compact'
out.mkdir(exist_ok=True)
target = root / 'Latex PDF/figures/fig_2_1.pdf'
fig.savefig(target, bbox_inches='tight', pad_inches=.03)
fig.savefig(out/'fig_2_1.png', dpi=180, bbox_inches='tight', pad_inches=.03)
plt.close(fig)
(out/'provenance.json').write_text(json.dumps({'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'output_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'scope': 'Frozen arrays redrawn only; scene coordinates and spectral curves unchanged'}, indent=2)+'\n', encoding='utf-8')
