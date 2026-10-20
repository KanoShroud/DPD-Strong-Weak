"""Audited saved-result export; no solver calls or GPU access.

Keeps old tables/figures untouched and writes a versioned paper asset bundle.
All methods use the new squared-distance assignment. Cost batches stay separate.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from relocated_evidence import local_path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
BASE = ROOT/'outputs_e2e/unified/e2e_g10/r4'
DEST = HERE/'data/assignment_cost_20261019'
FIG = HERE/'figures/assignment_cost_20261019'
SOURCE = ROOT/'统一模型代码/gates/g10/r4/paper_checks/assignment_metrics.py'
spec = importlib.util.spec_from_file_location('assignment_metrics', SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
PROVENANCE = {}
LABELS = {'AP_1_JOINT':'AP (M=1)', 'AP_2_JOINT':'AP (M=2)',
          'AP_3_JOINT':'AP (M=3)', 'AP_4_JOINT':'AP (M=4)', 'AP_5_JOINT':'MCJ-AP (M=5)',
          'PROFILE_DIAG':'VP-BNOMP', 'LS':'LS peaks', 'MUSIC':'WB-MUSIC',
          'MVDR':'MVDR-known', 'AP_FULL':'Full AP', 'BNOMP_REF':'BNOMP-REF',
          'MULTISTART_FIXED':'Multistart ML', 'GA':'GA-ML', 'SBL':'WB-SBL'}
ZH = {'AP_1_JOINT':'单候选联合AP', 'AP_2_JOINT':'双候选联合AP',
      'AP_3_JOINT':'三候选联合AP', 'AP_4_JOINT':'四候选联合AP', 'AP_5_JOINT':'五候选联合AP',
      'PROFILE_DIAG':'VP-BNOMP', 'LS':'LS谱取峰', 'MUSIC':'宽带MUSIC', 'MVDR':'MVDR-known',
      'AP_FULL':'充分AP', 'BNOMP_REF':'固定系数BNOMP', 'MULTISTART_FIXED':'多初值联合ML',
      'GA':'GA联合ML', 'SBL':'宽带SBL'}
AP = tuple(f'AP_{i}_JOINT' for i in range(1,6))
PAN = ('LS','MUSIC','MVDR','AP_FULL','BNOMP_REF','PROFILE_DIAG',
       'MULTISTART_FIXED','GA','SBL','AP_5_JOINT')
COLORS = dict(zip(AP, ['#4DBBD5','#009E73','#AA4499','#DDAA33','#3C5488']))
COLORS.update(PROFILE_DIAG='#E64B35', LS='#666666', MUSIC='#00A087', MVDR='#7E6148',
              AP_FULL='#8491B4', BNOMP_REF='#B09C85', MULTISTART_FIXED='#925E9F', GA='#D59A35', SBL='#4DBBD5')
STYLES = dict(zip(AP, ['--',':','-.',(0,(5,1,1,1)),'-']))
STYLES.update(PROFILE_DIAG='-.', LS=':', MUSIC='--', MVDR='-.', AP_FULL='--',
              BNOMP_REF=':', MULTISTART_FIXED=(0,(5,1,1,1)), GA='-.', SBL=':')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p, expected=None):
    p = local_path(p)
    h = digest(p)
    if expected is not None and h != expected:
        raise ValueError('Evidence changed: '+str(p))
    PROVENANCE[str(p)] = h
    return json.loads(p.read_text(encoding='utf-8'))


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def csv_write(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def source(pointer_name):
    ptr = read(BASE/(pointer_name+'.json')); folder = local_path(ptr['out'])
    assert folder.is_relative_to(BASE.resolve())
    c = read(folder/'contract.json', ptr['contract_sha'])
    audit = read(folder/'final_audit_report.json')
    assert audit['passed']
    return folder, c, audit


def flat(row, method, batch):
    task = row.get('task', row.get('job', {}).get('task'))
    result = row['result']
    q = metrics.evaluate(row['truth'], result['positions'])
    assert q['matched'] == 2 and q['valid_output']
    out = dict(batch=batch, method=method, case='_'.join(map(str,task[:3])),
               layout=task[0], repeat=task[1], gap=task[2], strong_error_m=q['errors_m'][0],
               weak_error_m=q['errors_m'][1], strong20=q['success20'][0], weak20=q['success20'][1],
               both20=q['both20'], seconds=result['complete_seconds'], stop_reason=result['stop_reason'])
    for k in ('joint_evaluations','gradient_calls','frequency_response_points','iterations',
              'fixed_q_derivative_calls','profile_schur_calls'):
        out[k] = result['counts'].get(k, 0)
    return out


def summarize(rows):
    t=np.array([r['seconds'] for r in rows]); e=np.array([[r['strong_error_m'],r['weak_error_m']] for r in rows])
    return dict(n=len(rows), both_count=sum(r['both20'] for r in rows),
        both_percent=100*np.mean([r['both20'] for r in rows]),
        strong_count=sum(r['strong20'] for r in rows), weak_count=sum(r['weak20'] for r in rows),
        strong_rmse_m=float(np.sqrt(np.mean(e[:,0]**2))), weak_rmse_m=float(np.sqrt(np.mean(e[:,1]**2))),
        weak_median_m=float(np.median(e[:,1])), weak_p95_m=float(np.quantile(e[:,1],.95)),
        weak_p99_m=float(np.quantile(e[:,1],.99)), mean_seconds=float(t.mean()),
        median_seconds=float(np.median(t)), p95_seconds=float(np.quantile(t,.95)),
        joint_evaluations_mean=float(np.mean([r['joint_evaluations'] for r in rows])),
        gradient_calls_mean=float(np.mean([r['gradient_calls'] for r in rows])),
        iterations_mean=float(np.mean([r['iterations'] for r in rows])))


def summary_tables(rows, name, methods):
    summaries=[]
    for m in methods:
        for group, keep in [('all',lambda r:True),('other19',lambda r:r['layout']!=3),('layout4',lambda r:r['layout']==3)]:
            rr=[r for r in rows if r['method']==m and keep(r)]
            summaries.append(dict(method=m, group=group, **summarize(rr)))
    csv_write(DEST/(name+'_summary.csv'), summaries)
    head='方法 & 双源成功率/\\% & 强源RMSE & 弱源RMSE & 时间均值 & 中位数 & P95\\\\\\midrule\n'
    body=''
    for s in summaries:
        if s['group'] != 'all':continue
        body += (f"{ZH[s['method']]} & {s['both_percent']:.2f} ({s['both_count']}/{s['n']}) & "
                 f"{s['strong_rmse_m']:.3f} & {s['weak_rmse_m']:.3f} & {s['mean_seconds']:.4f} & "
                 f"{s['median_seconds']:.4f} & {s['p95_seconds']:.4f}\\\\\n")
    (DEST/(name+'_table_rows.tex')).write_text(head+body, encoding='utf-8')
    return summaries


def plot_ecdf(ax, values, label, color, style, xmax):
    values=np.sort(np.asarray(values)); y=np.arange(1,len(values)+1)/len(values)
    ax.step(np.r_[0, values, xmax], np.r_[0,y,1.], where='post', label=label, color=color, linestyle=style)
    ax.set_ylim(0,1.015); ax.set_ylabel('Empirical CDF'); ax.grid(alpha=.14)


def savefig(name, fig, rows, zh, en, scope):
    fig.savefig(FIG/(name+'.png'), dpi=600, bbox_inches='tight')
    with Image.open(FIG/(name+'.png')) as im:
        gray=im.convert('L'); gray.thumbnail((1600,1100)); gray.save(FIG/(name+'_grayscale.png'))
    plt.close(fig)
    csv_write(FIG/(name+'.csv'), rows)
    save_json(FIG/(name+'_caption.json'), dict(zh=zh,en=en,scope=scope,
              evaluation=metrics.REVISION, sources=list(PROVENANCE)))


def error_cdf(rows, name, methods):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8), layout='constrained')
    for col, field in enumerate(('strong_error_m','weak_error_m')):
        xmax=max(20.,max(r[field] for r in rows))*1.1
        for j,m in enumerate(methods):
            v=[r[field] for r in rows if r['method']==m]
            for row in (0,1):plot_ecdf(axes[row,col],v,LABELS[m],COLORS[m],STYLES[m],xmax)
        axes[0,col].set_xscale('symlog', linthresh=1)
        axes[0,col].set_xlim(0,xmax)
        axes[0,col].axvline(20,color='#888888',lw=.8,ls=':')
        axes[0,col].set_title(('(a) Strong source','(b) Weak source')[col])
        axes[1,col].set_xlim(0,5 if col==0 else 10)
        axes[1,col].set_title(('(c) Strong: small errors','(d) Weak: small errors')[col])
        for row in (0,1):axes[row,col].set_xlabel('Position error (m)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=min(5,len(methods)),frameon=False,fontsize=7)
    savefig(name,fig,rows,'强弱源定位误差CDF；上行为线性—对数坐标（1m以内线性），下行为线性坐标的小误差区间放大，所有面板均以全部样本为分母。',
            'Strong/weak position-error CDFs; top: symlog axes, linear within 1 m; bottom: linear zooms retaining the full-sample denominator.',
            dict(n_per_method=len(rows)//len(methods), all_layouts=True, methods=list(methods)))


def costs(rows, name, methods):
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    xmax=max(r['seconds'] for r in rows)*1.08
    for j,m in enumerate(methods):
        v=[r['seconds'] for r in rows if r['method']==m]
        for ax in axes:plot_ecdf(ax,v,LABELS[m],COLORS[m],STYLES[m],xmax)
    axes[0].set_xscale('symlog',linthresh=.1);axes[0].set_title('(a) Complete range')
    axes[0].set_xlim(0,xmax)
    axes[1].set_xlim(0,1.5);axes[1].set_title('(b) Early completion')
    for ax in axes:ax.set_xlabel('Complete algorithm time (s)')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=min(5,len(methods)),frameon=False,fontsize=7)
    savefig(name,fig,rows,'完整算法耗时CDF，包含预处理、搜索、精修和输出检查；左图横轴0.1s以内线性、其外对数，右图线性放大；公共预热单列。',
            'CDF of complete algorithm time; left: symlog axis, linear within 0.1 s; right: linear zoom; common warmup is separate.',dict(n_per_method=240, methods=list(methods)))


def paired_statistics(rows):
    methods=AP+('PROFILE_DIAG',)
    keys=sorted({r['case'] for r in rows},key=lambda k:tuple(map(int,k.split('_'))))
    maps={m:{r['case']:r for r in rows if r['method']==m} for m in methods}
    for m in methods:assert set(maps[m])==set(keys)
    # Fixed-layout inference: resample the two realizations within each layout,
    # retain all six power gaps, and use the same draw for every method.
    index={k:i for i,k in enumerate(keys)}
    clusters=np.array([[[index[f'{g}_{r}_{gap}'] for gap in (0,3,6,9,12,15)]
                         for r in (400,410)] for g in range(20)])
    rng=np.random.default_rng(2026101901)
    draw=rng.integers(0,2,size=(2000,20,2))
    chosen=clusters[np.arange(20)[None,:,None],draw].reshape(2000,240)
    confidence={}; arrays={}
    for m in methods:
        rr=[maps[m][k] for k in keys]
        a=np.array([[r['seconds'],float(r['both20']),r['weak_error_m']**2] for r in rr])
        arrays[m]=a
        v=a[chosen].mean(axis=1);v[:,1]*=100;v[:,2]=np.sqrt(v[:,2])
        confidence[m]={name:np.quantile(v[:,j],[.025,.975]).tolist()
                       for j,name in enumerate(('mean_seconds','both_percent','weak_rmse_m'))}
    pairs=[]
    for m in AP:
        a,b=arrays['PROFILE_DIAG'],arrays[m]
        both=(a[:,1]>0)&(b[:,1]>0)
        diff=a[:,0]-b[:,0]
        pairs.append(dict(ap=m,n=240, vp_faster=int(np.sum(diff<0)), both_success=int(both.sum()),
            vp_faster_both_success=int(np.sum((diff<0)&both)),
            mean_time_diff_vp_minus_ap=float(diff.mean()),
            paired_time_diff_ci=np.quantile(diff[chosen].mean(axis=1),[.025,.975]).tolist(),
            vp_only_success=int(np.sum((a[:,1]>0)&(b[:,1]==0))),
            ap_only_success=int(np.sum((b[:,1]>0)&(a[:,1]==0)))))
    save_json(DEST/'paired_analysis.json',dict(confidence=confidence,pairs=pairs,
        resampling='2000 paired draws, fixed 20 layouts, 2 realizations per layout; six gaps kept together; conditional not geometry-population inference'))
    return confidence,pairs


def tradeoff(rows, summaries, confidence):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
    plotrows=[]
    for j,m in enumerate(AP+('PROFILE_DIAG',)):
        s=next(s for s in summaries if s['method']==m and s['group']=='all')
        for col,(ax,field) in enumerate(zip(axes,('both_percent','weak_rmse_m'))):
            x,y=s['mean_seconds'],s[field]
            cx,cy=confidence[m]['mean_seconds'],confidence[m][field]
            ax.errorbar(x,y,xerr=[[max(0,x-cx[0])],[max(0,cx[1]-x)]],
                        yerr=[[max(0,y-cy[0])],[max(0,cy[1]-y)]],
                        color=COLORS[m],marker='s' if m=='PROFILE_DIAG' else 'o',capsize=2,lw=.8)
            offset = (3,7 if j%2==0 else -14)
            if col==0:
                offset=[(-12,10),(-12,10),(-12,10),(-30,10),(-20,-18),(10,22)][j]
            ax.annotate('VP' if m=='PROFILE_DIAG' else 'M='+m.split('_')[1],(x,y),
                        xytext=offset,textcoords='offset points',fontsize=8)
        plotrows.append(s)
    for ax in axes:ax.set_xlabel('Mean complete algorithm time (s)');ax.grid(alpha=.15)
    axes[0].set_ylabel('Both-source success (%)');axes[1].set_ylabel('Weak-source RMSE (m)')
    axes[0].set_ylim(94.4,100.5)
    axes[0].set_title('(a) Reliability and cost');axes[1].set_title('(b) Error and cost')
    savefig('candidate_tradeoff',fig,plotrows,
            '同一240条观测的AP候选数与VP比较。误差棒为固定布局内按随机实现配对重采样的95%区间。',
            'AP candidate-count / VP tradeoffs on the same 240 observations; 95% paired fixed-layout bootstrap intervals.',
            dict(n_per_method=240, all_layouts=True, native_stops=True))


def main():
    DEST.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.3,
        'pdf.fonttype':42,'axes.linewidth':.7})
    auditptr=read(BASE/'assignment_audit_run.json')
    replay=read(Path(auditptr['out'])/'audit.json',auditptr['report_sha'])
    assert replay['passed'] and replay['protocol']==metrics.REVISION
    formal,_,fa=source('formal_run');profile,_,pa=source('profile_full_run')
    new,nc,na=source('candidate_cost_run')
    assert read(new/'status.json')['status']=='COMPLETED' and na['count']==1440
    core=[];common=[];timing=[]
    for fname,h in fa['rows'].items():
        parts=fname[:-5].split('_');m='_'.join(parts[3:]);repeat=int(parts[1])
        usecore=m in ('AP_1_JOINT','AP_2_JOINT','AP_5_JOINT')
        usecommon=m in PAN and repeat in (400,410)
        if not usecore and not usecommon:continue
        row=read(formal/'rows'/fname,h)
        if usecore:core.append(flat(row,m,'core_original'))
        if usecommon:common.append(flat(row,m,'common_original'))
    for fname,h in pa['rows'].items():core.append(flat(read(profile/'rows'/fname,h),'PROFILE_DIAG','profile_full_separate'))
    for fname,h in na['rows'].items():timing.append(flat(read(new/'rows'/fname,h),read(new/'rows'/fname)['method'],'candidate_paired'))
    assert len(core)==9600 and len(common)==2400 and len(timing)==1440
    for rows,name in [(core,'core'),(common,'common'),(timing,'candidate')]:csv_write(DEST/(name+'_per_case.csv'),rows)
    summary_tables(core,'core',('AP_1_JOINT','AP_2_JOINT','AP_5_JOINT','PROFILE_DIAG'))
    summary_tables(common,'common',PAN)
    sums=summary_tables(timing,'candidate',AP+('PROFILE_DIAG',))
    confidence,pairs=paired_statistics(timing)
    error_cdf(core,'core_error_cdf',('AP_1_JOINT','AP_2_JOINT','AP_5_JOINT','PROFILE_DIAG'))
    error_cdf(common,'common_error_cdf',PAN)
    costs(common,'common_time_cdf',PAN)
    costs(timing,'candidate_time_cdf',AP+('PROFILE_DIAG',))
    tradeoff(timing,sums,confidence)
    # Full-2400 M=3/4 POSITION replay only; never invent their full-cohort times.
    replayed=[]
    for fname,h in fa['rows'].items():
        if not fname.endswith('_AP_5_JOINT.json'):continue
        row=read(formal/'rows'/fname,h)
        for n in (3,4):
            b=min(row['result']['mechanism']['branches'][:n],key=lambda x:(x['objective'],x['rank']))
            q=metrics.evaluate(row['truth'],b['positions'])
            replayed.append(dict(case='_'.join(map(str,row['job']['task'][:3])),method=f'AP_{n}_JOINT',
                strong_error_m=q['errors_m'][0],weak_error_m=q['errors_m'][1],both20=q['both20']))
    csv_write(DEST/'full_prefix_position_replay.csv',replayed)
    report=dict(protocol=metrics.REVISION, timing_run=str(new), method_summaries=[s for s in sums if s['group']=='all'],
        paired=pairs, audit_changes=len(replay['changes']), invalid_outputs=0,
        note='Common240 timing and full2400 position evidence kept separate; native solver stops unchanged.',
        source_files=PROVENANCE, code_sha256=digest(Path(__file__)), evaluator_sha256=digest(SOURCE))
    save_json(DEST/'report.json',report)
    text=['# AP候选数与VP共同计时回读','',f'统一评价：{metrics.REVISION}。共同240条，六版本、1440次求解，输入和历史位置复现审计通过。','',
          '|版本|双源成功|弱源RMSE/m|平均/s|中位/s|P95/s|','|---|---:|---:|---:|---:|---:|']
    for s in report['method_summaries']:
        text.append(f"|{s['method']}|{s['both_count']}/240|{s['weak_rmse_m']:.4f}|{s['mean_seconds']:.4f}|{s['median_seconds']:.4f}|{s['p95_seconds']:.4f}|")
    text+=['','所有计时均含预处理、搜索、精修与输出检查；不跨批计算提速。原生停止不同，结果属于整套冻结算法比较，不作为同优化精度的更新机制速度证明。',
           '三、四候选全2400条仅回放已保存分支的位置，不外推完整耗时；见full_prefix_position_replay.csv。',
           '全部布局均保留，分组仅用于解释，见各summary.csv。GOSPA历史数据保留，本文新图表不再包含。']
    (DEST/'回读分析.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(report['method_summaries'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
