"""Read saved SBL diagnostics and export paired, traceable development figures."""
import argparse
from pathlib import Path
import numpy as np
from .runtime import fresh, preflight, read, write, sha, finish_audit
from .diagnostic_report import checked_rows
from .report import export, csv_write, plt
from .config import BASE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    out = fresh('sbl_repair_report')
    preflight(out, dict(stage='SBL_REPAIR_DEVELOPMENT_REPORT', runs=[str(p) for p in args.runs],
                        inference_performed=False, formal_readiness=False))
    sources, rows, summaries = {str(Path(__file__)): sha(__file__)}, [], []
    for folder in args.runs:
        for name, r in checked_rows(folder).items():
            if r.get('method') != 'SBL':
                continue
            cfg, result = r['result']['config'], r['result']
            key = f'{folder.name}/{name}'
            sources[str(folder/'rows'/name)] = sha(folder/'rows'/name)
            item = dict(run=key, group=r['scene']['group'], realization=r['scene']['realization'],
                        kind=r['kind'], gap_db=r['scene']['gap_db'],
                        clean=bool(r.get('clean', r['kind'] == 'clean_weak')),
                        active_atoms=cfg.get('sbl_active_atoms', 0),
                        spectral_model=cfg.get('sbl_spectral_model', 'flat'),
                        errors_m=r['metrics']['errors_m'], success20=all(r['metrics']['success']['20']),
                        seconds=result['seconds'], iterations=result['counts']['iterations'],
                        stop=result['stop_reason'], input_sha=r['input_sha'])
            summaries.append(item); rows.append((item, r))
    csv_write(out/'per_case.csv', summaries)
    write(out/'report.json', dict(scope='DEVELOPMENT_NOT_FORMAL_RANKING', rows=summaries,
                                  sources=sources, formal_ready=False))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    data = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    styles = [('#0072B2', '--'), ('#D55E00', '-'), ('#009E73', '-.'), ('#CC79A7', ':')]
    for clean, ax in zip((True, False), axes):
        subset = [(s, r) for s, r in rows if s['group'] == 3 and s['realization'] == 0
                  and s['clean'] == clean and len(s['errors_m']) == 1]
        for i, (s, r) in enumerate(subset):
            h = r['result']['history']; truth = np.array(r['truth'])[0]
            ii = [v['iteration'] for v in h]
            ee = [float(np.linalg.norm(np.array(v['positions'])[0]-truth)) for v in h]
            vv = [v['native_objective'] for v in h]
            color, line = styles[i % len(styles)]
            label = 'K decoded atoms' if not s['active_atoms'] else f"{s['active_atoms']} active atoms"
            ax[0].plot(ii, ee, color=color, ls=line, label=label)
            ax[1].plot(ii, vv, color=color, ls=line, label=label)
            data.extend(dict(run=s['run'], clean=clean, iteration=j, error_m=e, evidence=v)
                        for j, e, v in zip(ii, ee, vv))
        ax[0].axhline(20, color='black', ls=':', lw=.8)
        ax[0].set(xlabel='Iteration', ylabel='Position error (m)',
                  title='Noise removed, regularizer retained' if clean else 'Original noisy observation')
        ax[1].set(xlabel='Iteration', ylabel='Native negative log evidence')
        if subset:
            ax[0].legend(fontsize=7)
    export(out, '01_paired_trajectories', fig, data,
           '布局3、实现0的SBL适配轨迹。上下分别为删除噪声但保留原正则、正常有噪声观测；同条件输入字节一致，真值只用于画误差。不同曲线对应各自登记配置；没有根据最小误差挑选停止轮次。',
           'Paired layout-3 realization-0 SBL diagnostics. Noise-removed stress test and original noisy data; truth used only for error display. Native stopping, no truth-selected epoch.',
           sources, 'SBL ADAPTATION / DEVELOPMENT DIAGNOSTIC')
    forensic = BASE/'sbl_forensic_20261014_142609/report.json'
    if forensic.exists():
        raw = read(forensic)['rows']; sources[str(forensic)] = sha(forensic)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        for clean, ax in zip((True, False), axes):
            for radius, color, marker, line in [(20, '#0072B2', 'o', '-'), (100, '#D55E00', 's', '--')]:
                rr = [r for r in raw if r['clean'] == clean and r['radius_m'] == radius]
                ax.plot([r['active_count'] for r in rr], [r['error_of_highest_gamma_m'] for r in rr],
                        color=color, marker=marker, ls=line, label=f'{radius} m bounds')
            ax.set(xlabel='Number of movable atoms', ylabel='Highest-gamma atom error (m)',
                   xticks=[1, 2, 4, 8], title='Noise removed' if clean else 'Noisy')
            ax.legend(fontsize=8)
        export(out, '02_fixed_state_cause_test', fig, raw,
               '固定旧SBL末态、全部γ和原生目标，只改变允许移动的字典点数及移动范围。每条曲线为一个开发观测，不代表总体成功率。纵轴是最大γ原子的误差，不是完整求解器重跑结果。',
               'Fixed-state native-objective cause test: only movable atoms and bounds vary. One development observation per panel, not a population success rate or a from-scratch run.',
               sources, 'SBL ADAPTATION / CONTROLLED CAUSE TEST')
    completed = [(s, r) for s, r in rows if s['active_atoms'] == 8 and len(s['errors_m']) == 1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    displayed = []
    for clean, ax in zip((True, False), axes):
        rr = [s for s, r in completed if s['clean'] == clean]
        for s in rr:
            error = s['errors_m'][0]
            ax.scatter(s['group'], error, marker='o', color='#0072B2', s=35)
            ax.annotate(f'{error:.2f}', (s['group'], error), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)
            displayed.append(dict(group=s['group'], clean=clean, error_m=error,
                                  run=s['run'], seconds=s['seconds'], stop=s['stop']))
        ax.axhline(20, color='black', ls='--', lw=.8, label='20 m criterion')
        ax.set(xlabel='Fixed layout', ylabel='Position error (m)', xticks=range(4),
               xlim=(-.3, 3.3), ylim=(0, max(24, max((s['errors_m'][0] for s in rr), default=0)*1.2)),
               title='Noise removed' if clean else 'Noisy')
        ax.legend(fontsize=8)
    export(out, '03_single_source_regression', fig, displayed,
           '8个活动字典点、平坦跨频先验的单源回归；每布局一次实现，左右各n=4。显示所有完成样本，含仍超20m的场外案例；不代表独立泛化成绩。',
           'Single-source regression with eight active atoms and the flat frequency prior; n=4 per panel, one realization per layout. All completed cases including the failure are shown; not independent generalization evidence.',
           sources, 'SBL ADAPTATION / DEVELOPMENT REGRESSION')
    finish_audit(out)
    print('REPORT='+str(out), flush=True)


if __name__ == '__main__':
    main()
