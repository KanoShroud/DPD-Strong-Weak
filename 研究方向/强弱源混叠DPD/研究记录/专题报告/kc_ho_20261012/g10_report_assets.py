"""Chinese report export: saved metrics and spatial arrays only; no solver calls."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
RUN = ROOT / 'outputs_e2e/unified/e2e_g10/formal_20261013_231138'
TEX = BASE / 'Latex PDF'
OUT = BASE / 'qa_g10_report'
OUT.mkdir(exist_ok=True)
METHODS = ['LS', 'MUSIC', 'MVDR', 'AP', 'BNOMP', 'SBL', 'MULTISTART', 'GA']
NAMES = ['LS双峰', '宽带MUSIC', '已知响应MVDR', 'AP', '块NOMP', 'SBL-PDL-known', '多初值联合优化', 'GA联合优化']
COLORS = ['#576879', '#879baf', '#8b795e', '#275c68', '#689482', '#a18391', '#a19b66', '#505050']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
LINES = ['-', '--', '-.', ':', '--', '-.', ':', '-']
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 9, 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig, name):
    fig.savefig(TEX / 'figures' / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(TEX / 'figures' / (name + '.png'), dpi=250, bbox_inches='tight')
    plt.close(fig)


def legend(fig, axes):
    h, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, labels, loc='upper center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .87))


def main():
    sys.path.insert(0, str(ROOT))
    # Pure matching/metrics routine; no simulator or estimator imported.
    from 统一模型代码.gates.g10.metrics import evaluate
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((RUN / 'rows').glob('*.json'))]
    frozen = json.loads((RUN / 'comparison_report.json').read_text(encoding='utf-8'))
    for name, digest in frozen['sources'].items():
        assert sha(RUN / name) == digest, name
    source = {str(p.relative_to(RUN)): sha(p) for p in (RUN / 'maps').glob('*')}
    source.update(frozen['sources'])
    flat = []
    for r in rows:
        for t in [5, 15, 30, 60]:
            if t > r['result']['budget']:
                continue
            events = [e for e in r['result']['events'] if e['seconds'] <= t]
            pred = events[-1]['positions'] if events else []
            met = evaluate(r['truth'], pred)
            flags = met['success']['20']
            flat.append(dict(tag=r['tag'], group=r['scene']['group'], method=r['method'], kind=r['kind'],
                             gap_db=r['scene']['gap_db'], budget=t, strong20=int(flags[0]), weak20=int(flags[-1]),
                             both20=int(all(flags)), strong_error=met['errors_m'][0], weak_error=met['errors_m'][-1],
                             gospa=met['gospa_m'], seconds=r['result']['seconds'], coverage=met['matched']/met['targets']))
    data = pd.DataFrame(flat)
    stored = pd.read_csv(RUN / 'per_case_metrics.csv')
    a = data.set_index(['tag', 'budget']).sort_index()
    b = stored.set_index(['tag', 'budget']).sort_index()
    for col in ['strong20', 'weak20', 'both20', 'strong_error', 'weak_error', 'gospa', 'seconds', 'coverage']:
        assert np.allclose(a[col], b[col], equal_nan=True), col
    data.to_csv(OUT / 'recomputed_metrics.csv', index=False, encoding='utf-8-sig')
    mixed = data[(data.kind == 'mixed') & (data.budget == 60)]
    assert len(rows) == 1248 and len(mixed) == 576 and mixed.coverage.min() == 1
    summary = []
    for method in METHODS:
        d = mixed[mixed.method == method]
        summary.append(dict(method=method, n=len(d), strong=int(d.strong20.sum()), weak=int(d.weak20.sum()),
                            both=int(d.both20.sum()), strong_rmse=np.sqrt(np.mean(d.strong_error**2)),
                            weak_rmse=np.sqrt(np.mean(d.weak_error**2)), gospa=d.gospa.mean(),
                            mean_seconds=d.seconds.mean(), median_seconds=d.seconds.median(),
                            p95_seconds=d.seconds.quantile(.95), weak_p95=d.weak_error.quantile(.95)))
    pd.DataFrame(summary).to_csv(OUT / 'aggregate.csv', index=False, encoding='utf-8-sig')
    table = [r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}', r'\caption{60 s节点的混合源定位结果（每种方法72条记录）}\label{tab:g10-summary}',
             r'\begin{tabular}{lrrrrrr}\toprule', r'方法 & 强源成功 & 弱源成功 & 双源成功 & 强源RMSE/m & 弱源RMSE/m & 均值耗时/s\\\midrule']
    for name, r in zip(NAMES, summary):
        table.append(f"{name} & {r['strong']} & {r['weak']} & {r['both']} & {r['strong_rmse']:.2f} & {r['weak_rmse']:.2f} & {r['mean_seconds']:.2f}" + r'\\')
    table += [r'\bottomrule\end{tabular}', r'\end{table}']
    (TEX / 'g10_summary.tex').write_text('\n'.join(table), encoding='utf-8')
    cost = pd.read_csv(RUN / 'cost_table.csv').query("kind == 'mixed'").set_index('method')
    table = [r'\begin{table}[htbp]\centering\small', r'\caption{混合源完整耗时与平均运算量}\label{tab:g10-cost}',
             r'\begin{tabular}{lrrrrr}\toprule', r'方法 & 中位数/s & 95\%分位/s & 响应点数/$10^9$ & 联合评价次数 & 协方差分解次数\\\midrule']
    for name, r in zip(NAMES, summary):
        c = cost.loc[r['method']]
        table.append(f"{name} & {r['median_seconds']:.2f} & {r['p95_seconds']:.2f} & {c.mean_frequency_response_points/1e9:.3f} & {c.mean_joint_evaluations:.1f} & {c.mean_covariance_decompositions:.0f}" + r'\\')
    table += [r'\bottomrule\end{tabular}', r'\end{table}']
    (TEX / 'g10_cost.tex').write_text('\n'.join(table), encoding='utf-8')
    for stem, fields, time_axis in [
        ('g10_success', [('strong20','强源成功率 / %'),('weak20','弱源成功率 / %'),('both20','双源成功率 / %')], False),
        ('g10_error', [('strong_error','强源RMSE / m'),('weak_error','弱源RMSE / m'),('gospa','平均GOSPA / m')], False),
        ('g10_time', [('weak20','弱源成功率 / %'),('both20','双源成功率 / %')], True)]:
        fig, axes = plt.subplots(1, len(fields), figsize=(10.5, 3.8), squeeze=False)
        xx = [5,15,30,60] if time_axis else [0,3,6,9,12,15]
        for ax, (field, ylabel) in zip(axes.flat, fields):
            for i, method in enumerate(METHODS):
                vals = []
                for x in xx:
                    d = data[(data.kind == 'mixed') & (data.method == method) & (data.budget == x)] if time_axis else mixed[(mixed.method == method) & (mixed.gap_db == x)]
                    vals.append(np.sqrt(np.mean(d[field]**2)) if 'error' in field else d[field].mean()*(100 if field.endswith('20') else 1))
                ax.plot(xx, vals, color=COLORS[i], marker=MARKERS[i], linestyle=LINES[i], label=NAMES[i], ms=4, lw=1.2)
            ax.set(xlabel='时间节点 / s' if time_axis else '功率差 / dB', ylabel=ylabel, xticks=xx)
            if field.endswith('20'):
                ax.set_ylim(-3, 103)
            ax.grid(alpha=.18)
        legend(fig, axes)
        save(fig, stem)
    for target, label in [('strong','强源'),('weak','弱源')]:
        fig, axes = plt.subplots(2,3,figsize=(10.5,6.1))
        for ax, gap in zip(axes.flat, [0,3,6,9,12,15]):
            for i, method in enumerate(METHODS):
                values = np.sort(mixed[(mixed.method == method) & (mixed.gap_db == gap)][target+'_error'])
                ax.step(values, np.arange(1,len(values)+1)/len(values), where='post', color=COLORS[i], linestyle=LINES[i], marker=MARKERS[i], ms=3, label=NAMES[i], lw=1)
            ax.axvline(20,color='.7',lw=.8)
            ax.set_xscale('symlog',linthresh=20)
            ax.set(xlabel='定位误差 / m', ylabel='经验CDF', title=f'功率差 {gap} dB', xlim=(0,3000), ylim=(0,1.03))
            ax.set_xticks([0,20,100,1000], ['0','20','100','1000'])
            ax.grid(alpha=.18)
        legend(fig, axes)
        save(fig, 'g10_cdf_'+target)
    fig, ax = plt.subplots(figsize=(10.5,3.2))
    ax.boxplot([mixed[mixed.method == m].seconds for m in METHODS], tick_labels=NAMES, widths=.5,
               patch_artist=True, boxprops={'facecolor':'#d6dfe2'}, medianprops={'color':'#354b58'})
    ax.set(ylabel='完整处理时间 / s')
    ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    save(fig,'g10_runtime')
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.8),squeeze=False)
    for ax, target, kind in zip(axes.flat,['strong','weak'],['strong','weak']):
        for i, method in enumerate(METHODS):
            yy=[]
            for gap in [0,3,6,9,12,15]:
                iso=data[(data.kind==kind)&(data.budget==30)&(data.method==method)]
                if kind=='strong':
                    iso=iso[iso.gap_db==gap]
                mix=data[(data.kind=='mixed')&(data.budget==30)&(data.method==method)&(data.gap_db==gap)]
                yy.append(100*(iso[target+'20'].mean()-mix[target+'20'].mean()))
            ax.plot([0,3,6,9,12,15],yy,marker=MARKERS[i],linestyle=LINES[i],color=COLORS[i],label=NAMES[i],ms=4)
        ax.set(xlabel='功率差 / dB',ylabel=('强源' if target=='strong' else '弱源')+'成功率下降 / 百分点',xticks=[0,3,6,9,12,15])
        ax.axhline(0,color='.7',lw=.7)
        ax.grid(alpha=.18)
    legend(fig,axes)
    save(fig,'g10_paired')
    # Three paired snapshots, all eight methods, saved arrays only.
    for gap in [0,9,15]:
        fig, axes=plt.subplots(2,4,figsize=(10.8,5.8))
        for ax, method, name in zip(axes.flat,METHODS,NAMES):
            p=RUN/'maps'/f'0_{gap}_{method}.npz'
            z=np.load(p)
            points=z['points']; values=z['values'].reshape(-1)
            relative=(values-values.min())/max(float(np.ptp(values)),1e-30)
            sc=ax.scatter(points[:,0],points[:,1],c=relative,cmap='cividis',vmin=0,vmax=1,s=3,marker='s',rasterized=True)
            stations=np.array([[500,0],[0,500],[-500,0],[0,-500]])
            ax.scatter(stations[:,0],stations[:,1],marker='^',s=18,c='white',edgecolors='black',linewidths=.6)
            truth=z['truth']; pred=z['predicted']
            ax.scatter(truth[:,0],truth[:,1],marker='*',s=85,c='white',edgecolors='black',linewidths=.6)
            ax.scatter(pred[:,0],pred[:,1],marker='x',s=35,c='#e84a38',linewidths=1.2)
            meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
            if meta.get('fixed_position'):
                fixed=np.array(meta['fixed_position']).reshape(-1,2)
                ax.scatter(fixed[:,0],fixed[:,1],marker='o',s=85,facecolors='none',edgecolors='white',linewidths=1)
            ax.set(title=name,xlim=(-1000,1000),ylim=(-1000,1000),aspect='equal',xlabel='x / m',ylabel='y / m')
            ax.tick_params(labelsize=7)
        fig.subplots_adjust(left=.06,right=.91,bottom=.09,top=.94,wspace=.40,hspace=.45)
        cb=fig.colorbar(sc,cax=fig.add_axes([.93,.18,.012,.65]))
        cb.set_label('相对评分')
        save(fig,f'g10_maps_{gap}')
    result=dict(rows=len(rows), metric_records=len(data), metrics_match_frozen=True, sources=source,
                summary=summary, script_sha256=sha(Path(__file__)), operation='offline reaggregation and plotting only')
    (OUT/'data_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'metrics':len(data),'summary':summary},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
