"""冻结检查集报告；不根据检查成绩更换比较基线或距离门槛。"""
import numpy as np

from .metrics import aggregate
from .runtime import write


def finalize(out, rows, contract):
    from matplotlib import pyplot as plt
    summary = {}
    for n in (4096, 16384):
        for method in ('A', 'B', 'C', 'DPD', 'HR'):
            for factor in (.5, 1., 2.):
                subset = [r for r in rows if r['kind'] == 'mixed' and r['N'] == n
                          and r['method'] == method and r['factor'] == factor]
                summary[f'{n}_{method}_{factor}'] = aggregate(subset)
    comparisons = {}
    rng = np.random.default_rng(2026101009)
    for baseline in (contract['best_baseline'], 'B'):
        for factor in (.5, 1., 2.):
            differences = []
            for group in range(32):
                methods = {}
                for method in ('C', baseline):
                    subset = [r for r in rows if r['kind'] == 'mixed' and r['group'] == group
                              and r['method'] == method and r['factor'] == factor]
                    methods[method] = aggregate(subset)
                differences.append([methods['C'][key]-methods[baseline][key]
                                    for key in ('weak_100', 'both_100', 'strong_100')])
            d = np.asarray(differences)
            bootstrap = d[rng.integers(0, 32, (2000, 32))].mean(1)
            comparisons[f'C_minus_{baseline}_{factor}'] = dict(
                fields=['weak_100', 'both_100', 'strong_100'],
                mean=d.mean(0).tolist(), interval95=np.quantile(bootstrap, [.025, .975], axis=0).tolist())
    strata = {}
    for overlap in (.5, 1.):
        for gap in (6, 12):
            for method in ('A', 'B', 'C', 'DPD', 'HR'):
                strata[f'{overlap}_{gap}_{method}'] = aggregate([r for r in rows
                    if r['kind'] == 'mixed' and r['overlap'] == overlap and r['gap'] == gap
                    and r['method'] == method and r['factor'] == 1.])
    isolated = {}
    for n in (4096, 16384):
        for method in ('A', 'B', 'C', 'DPD', 'HR'):
            for kind in ('strong', 'weak'):
                subset = [r for r in rows if r['kind'] == kind and r['method'] == method
                          and r['N'] == n and r['factor'] == 1.]
                isolated[f'{n}_{method}_{kind}'] = dict(cases=len(subset),
                    recall100=float(np.mean([all(r['metrics']['success']['100']) for r in subset])),
                    rmse_m=aggregate(subset)['rmse_m'])
    speed = {}
    for n in (4096, 16384):
        for target in (.5, .7, .8, .9):
            for method in ('A', 'B', 'C', 'DPD', 'HR'):
                eligible = [summary[f'{n}_{method}_{f}']['seconds_mean'] for f in (.5, 1., 2.)
                            if summary[f'{n}_{method}_{f}']['weak_100'] >= target
                            and summary[f'{n}_{method}_{f}']['both_100'] >= target]
                speed[f'{n}_{target}_{method}'] = min(eligible) if eligible else None
    report = dict(status='COMPLETED_NEEDS_REVIEW', protocol='EVAL-20261010-v1',
        summary=summary, paired_comparisons=comparisons, strata=strata, isolated=isolated,
        time_to_reliability=speed, check_groups=32,
        decision='按预注册10pp/20%及强源2pp、尾部10%边界回读，不自动宣布论文结论',
        complexity=dict(A='O(Q*N*L + I*N*L*K^2)，含多初值和回访实际次数',
                        B='A加区域队列与细分；报告实际Q',
                        C='B加投影响应二维导数，O(Q_response*N*L*K)，不免费缓存迭代相关量',
                        DPD='O(NlogN + Q*N*L^2 + Q*L^3)',
                        HR='O(NlogN + (N/J)*(J*L^2+L^3) + Q*(N/J)*L^2 + Q*L^3)',
                        symbols='L站数，N点数，K源数，Q评分位置，I更新次数，J分段数；固定L=4/K=2不替代一般复杂度'),
        contract=contract)
    write(out/'comparison_report.json', report)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for row, n in enumerate((4096, 16384)):
        for method in ('A', 'B', 'C', 'DPD', 'HR'):
            values = [summary[f'{n}_{method}_{f}'] for f in (.5, 1., 2.)]
            for col, key in enumerate(('weak_100', 'both_100')):
                axes[row, col].plot([v['seconds_mean'] for v in values],
                                    [v[key] for v in values], marker='o', label=method)
                axes[row, col].set(xlabel='Complete online time (s)', ylabel=key,
                                   title=f'N={n}', ylim=(0, 1.02))
                axes[row, col].grid(alpha=.25)
                axes[row, col].legend()
    fig.savefig(out/'reliability_time.png', dpi=180)
    fig.savefig(out/'reliability_time.svg')
    plt.close(fig)
    (out/'运行摘要.md').write_text('# G9运行摘要\n\n状态：COMPLETED_NEEDS_REVIEW。\n\n'
        '请回读comparison_report.json、final_audit_report.json与evaluation_rows.json。\n'
        '曲线：reliability_time.png。未读取封存test。\n', encoding='utf-8')
    return report
