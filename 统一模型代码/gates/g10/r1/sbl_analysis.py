"""Offline paired output analysis; no truth enters decoding or solver selection."""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from ..metrics import evaluate
from .runtime import read, sha, write
from .sbl_outputs import VARIANTS, CHECKPOINTS

LABELS = ('Point gamma', 'Group gamma', 'Group coherent', 'Point power (aux)')
COLORS = ('#0072B2','#009E73','#D55E00','#666666')
LINES = ('-','--','-.',':')
MARKERS = ('o','s','^','x')


def read_replay(out, row):
    info = row['result']['shared_state']
    for field in ('manifest','replay'):
        if sha(out/info[field]) != info[field+'_sha']:
            raise RuntimeError('SBL output evidence changed: '+info[field])
    return read(out/info['replay'])


def expand_rows(out, rows):
    derived = []
    for r in rows:
        if r['method'] != 'SBL' or 'shared_state' not in r['result']:
            continue
        data = read_replay(out,r)
        state = data['states'][data['roles']['selected']]
        for variant in VARIANTS[1:3]:
            v = deepcopy(r)
            v['method'] = variant
            v['tag'] = r['tag']+'__'+variant
            v['derived_from'] = r['tag']
            v['result']['positions'] = state['outputs'][variant]
            v['result']['seconds'] = r['result']['seconds']+state['output_seconds'][variant]
            v['result']['events'] = []  # No invented continuous-time candidate availability.
            v['result']['stages'] = dict(r['result']['stages'],candidate_output=state['output_seconds'][variant])
            v['result']['counts']['candidate_extra_covariance_decompositions'] = state['extra_covariance_decompositions'] if variant==VARIANTS[2] else 0
            v['result']['counts']['candidate_extra_frequency_response_points'] = state['extra_frequency_response_points'] if variant==VARIANTS[2] else 0
            v['metrics'] = evaluate(r['truth'],v['result']['positions'])
            derived.append(v)
    return rows+derived


def paired_interval(rows):
    """Fixed-layout, realization-block pairing: all gaps stay together."""
    if not rows:
        return dict(mean=None,ci95=None,n=0)
    groups = sorted({r['group'] for r in rows})
    rng = np.random.default_rng(2026101402)
    means, boot = [], np.zeros(2000)
    valid = True
    for group in groups:
        sub = [r for r in rows if r['group']==group]
        reps = sorted({r['realization'] for r in sub})
        values = np.array([np.mean([r['difference'] for r in sub if r['realization']==rep]) for rep in reps])
        means.append(float(values.mean()))
        valid &= len(reps)>=2
        boot += values[rng.integers(len(values),size=(2000,len(values)))].mean(1)/len(groups)
    return dict(mean=float(np.mean(means)),ci95=np.quantile(boot,[.025,.975]).tolist() if valid else None,
                n=len(rows),layouts=len(groups),conditioning='Fixed geometries; paired realization blocks across gaps')


def generate(out, base_rows, source, label, export, csv_write):
    records, changed, first = [], [], []
    sbl = [r for r in base_rows if r['method']=='SBL' and 'shared_state' in r['result']]
    for row in sbl:
        saved = read_replay(out,row)
        source[row['result']['shared_state']['replay']] = row['result']['shared_state']['replay_sha']
        seen = []
        for role,key in saved['roles'].items():
            state = saved['states'][key]
            metrics = {v:evaluate(row['truth'],state['outputs'][v]) for v in VARIANTS}
            if role.isdigit() and state['duplicate_groups']:
                seen.append((int(role),state['iteration']))
            for variant in VARIANTS:
                metric = metrics[variant]
                errors = metric['errors_m']
                records.append(dict(tag=row['tag'],group=row['scene']['group'],realization=row['scene']['realization'],
                    gap_db=row['scene']['gap_db'],kind=row['kind'],role=role,state_iteration=state['iteration'],
                    carried_terminal=role.isdigit() and int(role)>state['iteration'],variant=variant,
                    both20=int(all(metric['success']['20'])),weak20=int(metric['success']['20'][-1]),
                    strong20=int(metric['success']['20'][0]),weak_error_m=errors[-1],strong_error_m=errors[0],
                    gospa_m=metric['gospa_m'],duplicate_groups=state['duplicate_groups'],
                    duplicate_atoms=state['duplicate_atoms'],native_objective=state['native_objective'],
                    solver_seconds=row['result']['seconds'] if role=='selected' else state['solver_seconds'],
                    output_seconds=state['output_seconds'][variant],
                    total_seconds=(row['result']['seconds'] if role=='selected' else state['solver_seconds'])
                                  +(state['output_seconds'][variant] if variant!='SBL' else 0.),
                    stop=row['result']['stop_reason'],state_sha=state['state_sha']))
            for variant,reference in ((VARIANTS[1],VARIANTS[0]),(VARIANTS[2],VARIANTS[1]),(VARIANTS[2],VARIANTS[3])):
                for target,index in (('both',None),('weak',-1),('strong',0)):
                    def success(v):
                        flags = metrics[v]['success']['20']
                        return bool(all(flags) if index is None else flags[index])
                    a,b = success(variant),success(reference)
                    changed.append(dict(tag=row['tag'],group=row['scene']['group'],realization=row['scene']['realization'],
                        gap_db=row['scene']['gap_db'],kind=row['kind'],role=role,variant=variant,reference=reference,
                        target=target,difference=int(a)-int(b),rescued=int(a and not b),harmed=int(b and not a),
                        triggered=bool(state['duplicate_groups'])))
        terminal = saved['states'][saved['roles']['terminal']]
        first.append(dict(tag=row['tag'],first_observed_checkpoint=min(seen)[0] if seen else None,
                          actual_state_iteration=min(seen)[1] if seen else None,
                          terminal_has_duplicate=bool(terminal['duplicate_groups']),
                          note='Observed at saved checkpoints only; not the exact first formation time'))
    if not records:
        return
    csv_write(out/'sbl_output_metrics.csv',records)
    csv_write(out/'sbl_output_changes.csv',changed)
    csv_write(out/'sbl_first_observed.csv',first)
    summaries = []
    for variant,reference in ((VARIANTS[1],VARIANTS[0]),(VARIANTS[2],VARIANTS[1]),(VARIANTS[2],VARIANTS[3])):
        for target in ('both','weak','strong'):
            for subset in ('all','duplicate_triggered'):
                rr = [r for r in changed if r['kind']=='mixed' and r['role']=='selected' and r['variant']==variant
                      and r['reference']==reference and r['target']==target and (subset=='all' or r['triggered'])]
                summaries.append(dict(variant=variant,reference=reference,target=target,subset=subset,
                    rescued=sum(r['rescued'] for r in rr),harmed=sum(r['harmed'] for r in rr),**paired_interval(rr)))
    report = dict(status='DEVELOPMENT' if 'DEVELOPMENT' in label else 'FORMAL_FIXED_LAYOUT_COMPARISON',
                  comparisons=summaries,solver_runs=len(sbl),sources=source,
                  iteration_cap=300,group_diameter_m=.05,variants=list(VARIANTS),
                  note='Three main outputs; point posterior power is an attribution-only control. No efficacy release threshold.')
    write(out/'sbl_comparison_report.json',report)
    csv_write(out/'sbl_comparison_table.csv',summaries)
    table = ['| Candidate | Reference | Target | Subset | N | Rescued | Harmed | Difference | 95% CI |',
             '|---|---|---|---|---:|---:|---:|---:|---|']
    tex = ['\\begin{tabular}{llllrrrr}','Candidate & Reference & Target & Subset & N & Rescued & Harmed & Difference \\\\']
    for r in summaries:
        table.append(f"| {r['variant']} | {r['reference']} | {r['target']} | {r['subset']} | {r['n']} | {r['rescued']} | {r['harmed']} | {r['mean']} | {r['ci95']} |")
        tex.append(' & '.join(str(r[k]).replace('_','\\_') for k in ('variant','reference','target','subset','n','rescued','harmed','mean'))+' \\\\')
    tex.append('\\end{tabular}')
    (out/'sbl_comparison_table.md').write_text('\n'.join(table),encoding='utf-8')
    (out/'sbl_comparison_table.tex').write_text('\n'.join(tex),encoding='utf-8')
    fig,axes = plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    for i,variant in enumerate(VARIANTS[:3]):
        for ax,field in zip(axes[:2],('weak20','both20')):
            vals=[]
            for step in CHECKPOINTS:
                rr=[r for r in records if r['kind']=='mixed' and r['role']==str(step) and r['variant']==variant]
                vals.append(np.mean([r[field] for r in rr]) if rr else np.nan)
            ax.plot(CHECKPOINTS,vals,color=COLORS[i],ls=LINES[i],marker=MARKERS[i],label=LABELS[i])
    yy=[]
    for step in CHECKPOINTS:
        rr=[r for r in records if r['variant']=='SBL' and r['role']==str(step)]
        yy.append(np.mean([r['duplicate_groups']>0 for r in rr]))
    axes[2].plot(CHECKPOINTS,yy,'o-',color='#666666')
    for ax,title in zip(axes,('Weak success @20m','Both success @20m','Any duplicate group')):
        ax.set(xlabel='Iteration budget',ylabel='Fraction',title=title,ylim=(-.03,1.03))
    axes[0].legend(fontsize=7)
    export(out,'sbl_01_iteration',fig,records,
           f'共同状态检查点比较；混合样本n={sum(r["kind"]=="mixed" for r in sbl)}，全部SBL样本n={len(sbl)}。提前停止后沿用终止态；开发短测不画无依据显著性。',
           'Shared-state checkpoints; terminal outputs carried after native stopping. Sample counts and paired intervals are in tables.',source,label)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),constrained_layout=True)
    chosen=[r for r in summaries if r['target']=='both' and r['subset']=='all']
    xx=np.arange(len(chosen))
    axes[0].bar(xx-.18,[r['rescued'] for r in chosen],width=.36,color='#009E73',hatch='//',label='Rescued')
    axes[0].bar(xx+.18,[r['harmed'] for r in chosen],width=.36,color='#D55E00',hatch='xx',label='Harmed')
    axes[0].set_xticks(xx,['Group gamma\nvs point gamma','Coherent\nvs group gamma','Coherent\nvs point power'])
    axes[0].set_ylabel('Mixed cases (both @20m)'); axes[0].legend(fontsize=7)
    axes[0].set_ylim(0,max(1,max([max(r['rescued'],r['harmed']) for r in chosen],default=0))*1.25)
    for i,r in enumerate(chosen):
        axes[0].text(i-.18,r['rescued']+.02,str(r['rescued']),ha='center',fontsize=8)
        axes[0].text(i+.18,r['harmed']+.02,str(r['harmed']),ha='center',fontsize=8)
    costs=[]
    for i,v in enumerate(VARIANTS[:3]):
        rr=[r for r in records if r['variant']==v and r['role']=='selected']
        base=float(np.mean([r['solver_seconds'] for r in rr])); extra=float(np.mean([r['output_seconds'] for r in rr])) if v!='SBL' else 0.
        axes[1].bar(i,base,color='#0072B2',label='Common solve' if i==0 else None)
        axes[1].bar(i,extra,bottom=base,color='#E69F00',hatch='//',label='Output processing' if i==0 else None)
        costs.append(dict(variant=v,mean_solver_seconds=base,mean_output_seconds=extra))
    axes[1].set_xticks(range(3),LABELS[:3],rotation=15); axes[1].set_ylabel('Mean complete algorithm time (s)');axes[1].legend(fontsize=7)
    export(out,'sbl_02_changes_cost',fig,costs,
           '全部混合样本的改善/损害计数及完整算法耗时；状态记录和离线研究重放成本另列。',
           'Rescues and harms on all mixed cases; common solve plus necessary output cost. Research recording overhead separate.',source,label)
    (out/'SBL输出分析.md').write_text('# SBL共同状态比较\n\n'+report['status']+'\n\n'
        '本报告不改变SBL内部优化。分组γ与单点γ比较分组作用；相干聚合另与单点后验功率比较，避免混淆评分变化。\n\n'
        +'\n'.join(table)+'\n\n首次重复仅指保存检查点观察到。详见sbl_output_metrics.csv和sbl_first_observed.csv。\n',encoding='utf-8')
