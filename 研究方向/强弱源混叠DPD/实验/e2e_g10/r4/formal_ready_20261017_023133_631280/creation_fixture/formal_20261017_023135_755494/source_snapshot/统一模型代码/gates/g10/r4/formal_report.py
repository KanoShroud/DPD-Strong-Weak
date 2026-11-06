"""Saved-result-only paired tables/figures; never invoke a localization solver."""
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..metrics import evaluate
from ..r2.report import csv_write, COLORS, MARKERS
from ...g8.physics import RECEIVERS
from .formal_protocol import CORE, METHODS, SUBSET_REPEATS, TIMES
from .runtime import read, write, sha, seal

# Coprime style/color periods keep ten panorama curves distinguishable in gray.
STYLES = ('-', '--', '-.', ':')


def flat(row,c):
    task=row['job']['task']; r=row['result']; metric=row['metrics']
    layout=c['layouts'][task[0]]
    return dict(layout=task[0],category=layout['category'],role=layout['role'],realization=task[1],
        gap_db=task[2],method=row['job']['method'],both20=int(all(metric['success']['20'])),
        strong20=int(metric['success']['20'][0]),weak20=int(metric['success']['20'][1]),
        strong_error_m=metric['errors_m'][0],weak_error_m=metric['errors_m'][1],
        gospa_m=metric['gospa_m'],seconds=r['complete_seconds'],stop=r['stop_reason'],
        projected_gradient=r['projected_gradient'],joint_objective=r['joint_objective'],
        **{'count_'+k:v for k,v in r['counts'].items()})


def paired_boot(items, field):
    """Stratify by geometry class; resample layouts then waveform realizations.

    Each repeat's complete power scan stays together. Same deterministic draws
    across methods permit paired differences; never treat six gaps as iid trials.
    """
    rng=np.random.default_rng(20261017044); boot=np.zeros(1000)
    cats=sorted({r['category'] for r in items})
    for cat in cats:
        sub=[r for r in items if r['category']==cat]
        layouts=sorted({r['layout'] for r in sub})
        values=[]
        for layout in layouts:
            here=[r for r in sub if r['layout']==layout]
            repeats=sorted({r['realization'] for r in here})
            values.append([np.mean([r[field] for r in here if r['realization']==rep]) for rep in repeats])
        if len({len(v) for v in values})!=1:
            raise ValueError('Unbalanced paired bootstrap input')
        values=np.asarray(values)
        li=rng.integers(len(layouts),size=(1000,len(layouts),1))
        ri=rng.integers(values.shape[1],size=(1000,len(layouts),values.shape[1]))
        boot+=values[li,ri].mean((1,2))/len(cats)
    return np.quantile(boot,[.025,.975]).tolist()


def summary(items):
    out=dict(n=len(items),both_count=sum(r['both20'] for r in items),
        both20=float(np.mean([r['both20'] for r in items])),both20_ci95=paired_boot(items,'both20'),
        mean_seconds=float(np.mean([r['seconds'] for r in items])),
        p95_seconds=float(np.quantile([r['seconds'] for r in items],.95)),
        gospa_m=float(np.mean([r['gospa_m'] for r in items])),
        stops=dict(Counter(r['stop'] for r in items)))
    for role in ('strong','weak'):
        errors=np.array([r[role+'_error_m'] for r in items])
        out.update({role+'20':float(np.mean([r[role+'20'] for r in items])),
                    role+'_rmse_m':float(np.sqrt(np.mean(errors**2))),
                    role+'_p95_m':float(np.quantile(errors,.95))})
    return out


def table(out,stem,rows):
    csv_write(out/(stem+'.csv'),rows)
    keys=('method','scope','n','both_count','both20','strong_rmse_m','weak_rmse_m','gospa_m','mean_seconds')
    cells=[[f'{r[k]:.5g}' if isinstance(r.get(k),float) else str(r.get(k,'')) for k in keys] for r in rows]
    lines=['| '+' | '.join(keys)+' |','|'+'|'.join(['---']*len(keys))+'|']
    lines+=['| '+' | '.join(row)+' |' for row in cells]
    (out/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tex='\\begin{tabular}{'+'l'*len(keys)+'}\n'
    tex+=' & '.join(k.replace('_','\\_') for k in keys)+' \\\\\n'
    tex+='\n'.join(' & '.join(x.replace('_','\\_') for x in row)+' \\\\' for row in cells)
    (out/(stem+'.tex')).write_text(tex+'\n\\end{tabular}\n',encoding='utf-8')


def export(folder,stem,fig,data,source,cn,en,label):
    folder.mkdir(parents=True,exist_ok=True)
    fig.suptitle(label,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('pdf','png'):
        fig.savefig(folder/(stem+'.'+ext),dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    csv_write(folder/(stem+'.csv'),data)
    write(folder/(stem+'.json'),dict(caption_cn=cn,caption_en=en,source_manifest=source,
        evaluation='EVAL-20261011-v2',plot_code_sha=sha(__file__),sample_selection=label))
    (folder/(stem+'_caption.md')).write_text(cn+'\n\n'+en+'\n',encoding='utf-8')


def plots(out,label,items,rows,methods,source):
    folder=out/'figures'; data=[]
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for i,method in enumerate(methods):
        part=[r for r in items if r['method']==method]
        for ax,target in zip(axes,('strong20','weak20','both20')):
            points=[]
            for gap in sorted({r['gap_db'] for r in part}):
                group=[r for r in part if r['gap_db']==gap]
                mean=float(np.mean([r[target] for r in group])); ci=paired_boot(group,target)
                points.append((gap,mean,ci)); data.append(dict(method=method,gap_db=gap,target=target,
                    value=mean,low=ci[0],high=ci[1],n=len(group)))
            ax.errorbar([p[0] for p in points],[p[1] for p in points],
                yerr=np.array([[max(0,p[1]-p[2][0]),max(0,p[2][1]-p[1])] for p in points]).T,
                color=COLORS[i%len(COLORS)],marker=MARKERS[i%len(MARKERS)],ls=STYLES[i%len(STYLES)],
                label=method,capsize=2,lw=1,ms=3)
            ax.set(xlabel='Power gap (dB)',ylabel='Success probability',title=target,ylim=(-.03,1.03))
    axes[-1].legend(fontsize=5.5,ncol=2)
    export(folder,'success',fig,data,source,'20 m成功率；误差条为按布局及随机实现分层配对重采样的95%区间。全成功时区间可能退化，不表示总体概率为1。',
        '20 m success; 95% hierarchical paired bootstrap intervals. Degenerate all-success intervals do not imply population certainty.',label)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5)); data=[]
    for i,method in enumerate(methods):
        part=[r for r in items if r['method']==method]
        for ax,target in zip(axes,('strong_error_m','weak_error_m','gospa_m')):
            points=[]
            for gap in sorted({r['gap_db'] for r in part}):
                group=[r for r in part if r['gap_db']==gap]
                v=np.array([r[target] for r in group]); value=float(v.mean() if target=='gospa_m' else np.sqrt(np.mean(v*v)))
                points.append((gap,value)); data.append(dict(method=method,gap_db=gap,metric=target,value=value,n=len(group)))
            ax.plot(*np.array(points).T,color=COLORS[i%len(COLORS)],marker=MARKERS[i%len(MARKERS)],
                    ls=STYLES[i%len(STYLES)],label=method,ms=3,lw=1)
            ax.set(xlabel='Power gap (dB)',ylabel='Error (m)',title='Mean GOSPA' if target=='gospa_m' else target.replace('_error_m',' RMSE'))
    axes[-1].legend(fontsize=5.5,ncol=2)
    export(folder,'errors',fig,data,source,'未截断强弱源RMSE及平均GOSPA，使用相同配对观测。','Untruncated source RMSE and mean GOSPA on matched observations.',label)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5)); data=[]
    for i,method in enumerate(methods):
        part=[r for r in items if r['method']==method]
        for ax,role in zip(axes,('strong','weak')):
            x=np.sort([r[role+'_error_m'] for r in part]); y=np.arange(1,len(x)+1)/len(x)
            ax.plot(x,y,color=COLORS[i%len(COLORS)],ls=STYLES[i%len(STYLES)],label=method)
            ax.set(xlabel='Position error (m)',ylabel='Empirical CDF',title=role,xscale='symlog')
            data.extend(dict(method=method,role=role,error_m=float(a),cdf=float(b)) for a,b in zip(x,y))
    axes[-1].legend(fontsize=5.5,ncol=2)
    export(folder,'cdf',fig,data,source,'全部误差均进入CDF，不删除长尾；横轴为对称对数尺度。','All errors retained; symlog error axis.',label)
    fig,ax=plt.subplots(figsize=(7,3.8)); data=[]
    keys={(r['layout'],r['realization'],r['gap_db'],r['method']) for r in items}
    for i,method in enumerate(methods):
        raw=[r for r in rows if r['job']['method']==method and tuple(r['job']['task'][:3])+ (method,) in keys]
        points=[]
        for seconds in TIMES:
            success=[]
            for row in raw:
                events=[e for e in row['result']['report_events'] if e['seconds']<=seconds]
                positions=events[-1]['positions'] if events else []
                success.append(all(evaluate(row['truth'],positions)['success']['20']))
            rate=float(np.mean(success)); points.append((seconds,rate))
            data.append(dict(method=method,seconds=seconds,both20=rate,n=len(raw)))
        ax.plot(*np.array(points).T,color=COLORS[i%len(COLORS)],marker=MARKERS[i%len(MARKERS)],
                ls=STYLES[i%len(STYLES)],label=method,ms=3)
    ax.set(xlabel='Elapsed time (s)',ylabel='Two-source success probability',xscale='log',ylim=(-.03,1.03))
    ax.legend(fontsize=6,ncol=3)
    export(folder,'anytime',fig,data,source,'按实际原生目标选中的早期输出；时间保守计入全部记录与证书开销。无完整双源输出时计未成功。',
        'Native-selected incumbents with conservatively charged total recording/certificate overhead; missing complete pairs count as unsuccessful.',label)


def case_maps(folder,rows,source,preparation):
    """Display only actually saved native scores; do not synthesize ML slices."""
    cases=sorted({tuple(r['job']['task']) for r in rows if r['result'].get('native_map')})
    for task in cases:
        raw=[r for r in rows if tuple(r['job']['task'])==task and r['result'].get('native_map')
             and r['job']['method'] in ('LS','MUSIC','MVDR','SBL')]
        if not raw:
            continue
        fig,axes=plt.subplots(1,len(raw),figsize=(3*len(raw),3.5),squeeze=False); data=[]
        for ax,row in zip(axes[0],raw):
            native=row['result']['native_map']; method=row['job']['method']
            pp=np.asarray(native['points'])
            statistic='gamma' if method=='SBL' and native.get('output_statistic')=='gamma' else 'power'
            value=np.asarray(native[statistic]); vmin,vmax=float(value.min()),float(value.max())
            shown=(value-vmin)/max(vmax-vmin,1e-30)
            ax.scatter(pp[:,0],pp[:,1],c=shown,cmap='viridis',s=3,vmin=0,vmax=1,rasterized=True)
            truth=np.asarray(row['truth']); pred=np.asarray(row['result']['positions'])
            ax.scatter(truth[:,0],truth[:,1],s=60,facecolors='none',edgecolors='red',marker='o',label='True')
            ax.scatter(pred[:,0],pred[:,1],s=35,c='white',marker='x',label='Estimated')
            ax.scatter(RECEIVERS[:,0],RECEIVERS[:,1],s=25,c='white',edgecolors='black',marker='^')
            ax.set(xlabel='x (m)',ylabel='y (m)',title=method+(' / gamma' if statistic=='gamma' else ''),
                xlim=(-1000,1000),ylim=(-1000,1000),aspect='equal')
            ax.legend(fontsize=6,loc='lower right')
            data.extend(dict(method=method,statistic=statistic,x_m=float(p[0]),y_m=float(p[1]),score=float(v),display=float(s))
                        for p,v,s in zip(pp,value,shown))
            for kind,points in (('truth',truth),('estimate',pred)):
                data.extend(dict(method=method,statistic=kind,x_m=float(p[0]),y_m=float(p[1]),score=None,display=None) for p in points)
        label=('DEVELOPMENT CHECK / ' if preparation else 'Preregistered case / ')+f'layout {task[0]}, realization {task[1]}, gap {task[2]} dB'
        export(folder,f'case_{task[0]}_{task[1]}_{task[2]}',fig,data,source,
            '预先指定案例的实际原生评分；各图独立归一化到0—1，不跨方法比较谱幅度。圆圈为真值，叉号为估计。',
            'Saved native scores for preregistered cases; each map is independently normalized to [0,1]. Circles: truth; crosses: estimates.',label)


def generate(out,c,stages,preparation=False):
    out=Path(out)
    folder=out/'reports'/('preparation' if preparation else stages[-1]); folder.mkdir(parents=True,exist_ok=True)
    rows=[]; identities={}
    for path in sorted((out/'rows').glob('*.json')):
        r=read(path)
        if r.get('job',{}).get('stage') not in stages:
            continue
        if r.get('record_sha')!=seal(r) or r['metrics']!=evaluate(r['truth'],r['result']['positions']):
            raise RuntimeError('Saved report row invalid')
        identities[str(path.relative_to(out))]=sha(path); rows.append(r)
    if not rows:
        raise RuntimeError('No report rows')
    write(folder/'source_manifest.json',identities)
    flat_rows=[flat(r,c) for r in rows]; csv_write(folder/'per_case_metrics.csv',flat_rows)
    scopes={'core': [r for r in flat_rows if r['method'] in CORE],
            'panorama': [r for r in flat_rows if r['realization'] in SUBSET_REPEATS]}
    if preparation:
        scopes={'core':[r for r in flat_rows if r['method'] in CORE], 'panorama':flat_rows}
    all_summaries=[]; paired=[]
    for scope,base in scopes.items():
        for role in ('all','analysis','locked_validation'):
            part=[r for r in base if role=='all' or r['role']==role]
            methods=[m for m in METHODS if any(r['method']==m for r in part)]
            if not part:
                continue
            # Only complete identical sample sets share a ranking.
            sets={m:{(r['layout'],r['realization'],r['gap_db']) for r in part if r['method']==m} for m in methods}
            if len({frozenset(v) for v in sets.values()})!=1:
                raise RuntimeError('Unpaired/incomplete methods in comparison scope')
            all_summaries.extend(dict(method=m,scope=scope+'/'+role,**summary([r for r in part if r['method']==m])) for m in methods)
            for left,right in (('AP_1_ALT','AP_1_JOINT'),('AP_5_JOINT','AP_2_JOINT'),
                               ('NOMP','NOMP_JOINT'),('BNOMP_REF','NOMP'),('BNOMP_REF','PROFILE_DIAG')):
                if left not in methods or right not in methods:
                    continue
                a={(r['layout'],r['realization'],r['gap_db']):r for r in part if r['method']==left}
                diffs=[]
                for r in part:
                    if r['method']!=right:
                        continue
                    v=a[r['layout'],r['realization'],r['gap_db']]
                    diffs.append(dict(r, delta_success=r['both20']-v['both20'],delta_seconds=r['seconds']-v['seconds']))
                paired.append(dict(scope=scope+'/'+role,left=left,right=right,n=len(diffs),
                    success_change=float(np.mean([r['delta_success'] for r in diffs])),
                    success_change_ci95=paired_boot(diffs,'delta_success'),
                    mean_seconds_change=float(np.mean([r['delta_seconds'] for r in diffs])),
                    seconds_change_ci95=paired_boot(diffs,'delta_seconds')))
            if role in ('all','locked_validation') and (not preparation or role=='all'):
                p=folder/(scope+'_'+role); p.mkdir(exist_ok=True)
                # The panorama figure includes eight baseline families plus two
                # predeclared candidate representatives. All versions stay in tables.
                shown=methods if scope=='core' else [m for m in methods if m not in ('NOMP_JOINT','AP_1_ALT','AP_1_JOINT','AP_5_JOINT','PROFILE_DIAG')]
                label=('DEVELOPMENT ENGINEERING CHECK / ' if preparation else 'G10-R4 / ')+scope+'/'+role+f' / n={len(sets[methods[0]])}'
                style=Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
                if style.exists():
                    plt.style.use(str(style))
                plt.rcParams.update({'figure.constrained_layout.use':False,'font.size':8,'legend.fontsize':6})
                plots(p,label,part,rows,shown,'../../source_manifest.json')
    table(folder,'summary_table',all_summaries); csv_write(folder/'paired_comparison.csv',paired)
    # Cost/complexity fields are exact per-case counters, not proxy wall time.
    costs=[]
    for s in all_summaries:
        scope,role=s['scope'].split('/')
        part=[r for r in scopes[scope] if r['method']==s['method'] and (role=='all' or r['role']==role)]
        counts={k:float(np.mean([r.get(k,0) for r in part])) for k in set().union(*(r.keys() for r in part)) if k.startswith('count_')}
        costs.append(dict(method=s['method'],scope=s['scope'],n=len(part),mean_seconds=s['mean_seconds'],**counts))
    csv_write(folder/'operation_counts.csv',costs)
    report=dict(stage='preparation' if preparation else stages[-1],completed=len(rows),summaries=all_summaries,paired=paired,
        no_cross_scope_ranking=True,bootstrap='1000 paired resamples: geometry class -> layouts -> realizations, retaining all power gaps',
        source_manifest='source_manifest.json',zero_gap_labels='scan identities, equal mean received power')
    write(folder/'comparison_report.json',report)
    (folder/'运行摘要.md').write_text('# G10-R4运行摘要\n\n'+('开发接入检查，不是正式性能证据。' if preparation else '本阶段已完成，完整方法排名仅比较相同观测。')+
        '\n\n统计见 summary_table.md；配对差异见 paired_comparison.csv；运算量见 operation_counts.csv；图表见 core_all、panorama_all及锁定验证子目录。\n',encoding='utf-8')
    # Saved geometry provides the predeclared scenario overview without rescoring.
    fig,ax=plt.subplots(figsize=(5,5)); data=[]
    ax.scatter(RECEIVERS[:,0],RECEIVERS[:,1],marker='^',c='black',label='Receivers')
    for layout in c['layouts']:
        p=np.array(layout['positions']); ax.plot(p[:,0],p[:,1],lw=.6,alpha=.5)
        ax.scatter(p[0,0],p[0,1],marker='o',s=12); ax.scatter(p[1,0],p[1,1],marker='x',s=16)
        data.extend(dict(layout=layout['id'],role=layout['role'],source=i,x_m=float(x),y_m=float(y)) for i,(x,y) in enumerate(p))
    ax.set(xlabel='x (m)',ylabel='y (m)',xlim=(-1000,1000),ylim=(-1000,1000),aspect='equal')
    ax.legend()
    export(folder/'figures','layouts',fig,data,'../source_manifest.json','预先固定布局；圆点为功率扫描源，叉号为固定弱源。',
        'Preregistered geometry; circles: power-scanned source; crosses: fixed weak source.', 'Preparation geometry' if preparation else 'G10-R4 geometry')
    case_maps(folder/'figures',rows,'../source_manifest.json',preparation)
    return report
