"""Offline evidence illustration; no new localization or parameter selection."""
import argparse
from pathlib import Path
import numpy as np
from .runtime import fresh, preflight, read, write, sha, finish_audit
from .report import export, csv_write, plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', required=True, type=Path)
    parser.add_argument('--short', required=True, type=Path)
    args = parser.parse_args()
    check, short = args.check.resolve(strict=True), args.short.resolve(strict=True)
    old = read(check/'contract.json')['input_identity']
    for path, expected in old.items():
        if sha(path) != expected:
            raise RuntimeError('Old evidence changed')
    audit = read(short/'final_audit_report.json')
    for name, expected in audit['rows'].items():
        if sha(short/'rows'/name) != expected:
            raise RuntimeError('Short row changed')
    native_path = next(Path(p) for p in old if '/maps/' in p.replace('\\', '/'))
    row_path = next(Path(p) for p in old if '/rows/' in p.replace('\\', '/'))
    native, row, result = read(native_path), read(row_path), read(check/'report.json')
    inputs = {str(check/'report.json'): sha(check/'report.json'),
              str(short/'summary.json'): sha(short/'summary.json'), **old}
    out = fresh('output_summary')
    preflight(out, dict(stage='OFFLINE_DEVELOPMENT_FIGURE', inputs=inputs,
                        no_inference=True, posthoc_failure_case=True))
    print('OUT='+str(out), flush=True)
    style = Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
    if style.exists():
        plt.style.use(style)
    plt.rcParams.update({'font.size':9, 'axes.labelsize':9, 'pdf.fonttype':42})
    pp, power = np.asarray(native['points']), np.asarray(native['power'])
    selected = np.argsort(-power)[:12]
    truth = np.asarray(row['truth'])[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.scatter(pp[selected, 0], pp[selected, 1], s=40+120*power[selected]/power.max(),
               facecolor='#56B4E9', edgecolor='black', linewidth=.6, label='Dictionary atoms')
    ax.scatter(*truth, marker='*', s=130, color='black', label='True source')
    peak = np.asarray(result['outputs'][0]['positions'])[0]
    merged = np.asarray(result['outputs'][2]['positions'])[0]
    ax.scatter(*peak, marker='x', s=90, color='#D55E00', linewidth=2, label='Point selection')
    ax.scatter(*merged, marker='s', s=100, facecolor='none', edgecolor='#009E73', linewidth=2, label='Group selection')
    ax.annotate('3 nearly coincident\natoms', xy=merged, xytext=(-25, -45), textcoords='offset points',
                fontsize=8, arrowprops={'arrowstyle':'->', 'lw':.7})
    ax.set(xlabel='x (m)', ylabel='y (m)', title='A  Saved posterior: local view', aspect='equal')
    focus = np.stack([truth, peak, merged])
    center = (focus.min(0)+focus.max(0))/2
    span = max(np.ptp(focus, axis=0).max()+60, 100.)
    ax.set(xlim=(center[0]-span/2, center[0]+span/2),
           ylim=(center[1]-span/2, center[1]+span/2))
    ax.legend(fontsize=7, loc='upper left')
    labels = ['Point posterior power', 'Point gamma (literature)', 'Group coherent power', 'Group gamma']
    errors = [r['metrics']['errors_m'][0] for r in result['outputs']]
    bars = axes[1].barh(labels, errors, color=['#56B4E9','#0072B2','#009E73','#E69F00'], edgecolor='black', linewidth=.5)
    for bar, hatch in zip(bars, ['', '//', 'xx', '..']):
        bar.set_hatch(hatch)
    axes[1].axvline(20, color='black', ls='--', lw=.8, label='20 m evaluation threshold')
    axes[1].bar_label(bars, labels=[f'{e:.2f}' for e in errors], padding=3, fontsize=8)
    axes[1].set(xlabel='Localization error (m)', xlim=(0, max(errors)*1.2),
                title='B  Same saved state, different output')
    axes[1].invert_yaxis(); axes[1].legend(fontsize=7, loc='lower right')
    data = [dict(type='selected_atom', id=int(i), x_m=float(pp[i,0]), y_m=float(pp[i,1]), power=float(power[i])) for i in selected]
    data += [dict(type='output', name=r['name'], x_m=r['positions'][0][0], y_m=r['positions'][0][1],
                  error_m=r['metrics']['errors_m'][0]) for r in result['outputs']]
    export(out, '01_output_consistency', fig, data,
           '历史100MHz单例事后诊断（n=1），非50MHz性能比较。固定字典、后验和0.05m分组直径，不重跑优化；聚合仅为候选。20m为既定评价门槛，不用于选择分组。无总体置信区间或显著性推断。',
           'Post-hoc historical 100 MHz case (n=1), not a 50 MHz benchmark. Fixed posterior and 0.05 m group diameter; no optimization rerun. Grouping is a candidate, not the default baseline. No population inference.',
           {**inputs, str(Path(__file__).resolve()):sha(__file__)}, 'DEVELOPMENT DIAGNOSTIC / n=1')
    summary = read(short/'summary.json')
    csv_write(out/'sampling_short_rows.csv', summary['rows'])
    write(out/'report.json', dict(output_check=str(check), short_run=str(short),
          figure='figures/01_output_consistency.png', inputs=inputs, formal_ready=False))
    incomplete = [r['tag'] for r in summary['rows'] if r['stop'] == 'ITERATION_LIMIT']
    text = (f'# 有限核验摘要\n\nSBL历史保存状态：逐点误差{errors[0]:.2f}m，近重复组输出{errors[2]:.2f}m。'
            '仅单例，不代表总体提升。\n\n'
            f"新采样配置：{summary['sample_rate_hz']/1e6:g}MHz/{summary['N']}点，{len(summary['rows'])}条短测完成。"
            f'迭代上限记录：{incomplete}。保留未确认收敛状态，不因定位误差继续调参。'
            '完整六档比较未放行，详细停止与成本见sampling_short_rows.csv。\n\n'
            '图示为100MHz历史案例，不能与50MHz短测混用。\n')
    (out/'运行摘要.md').write_text(text, encoding='utf-8')
    finish_audit(out)
    print('DONE '+str(out), flush=True)


if __name__ == '__main__':
    main()
