"""Read-only evidence analysis and figures for R1 qualification, never a ranking."""
import argparse
from pathlib import Path
import numpy as np
from .runtime import fresh, preflight, read, write, sha
from .report import export, csv_write, plt
from .config import METHODS


def checked_rows(root):
    audit = read(root/'final_audit_report.json')
    for name, expected in audit['inputs'].items():
        info = read(root/'inputs'/name)
        if sha(root/info['path']) != expected:
            raise RuntimeError('Saved input changed: '+name)
    rows = {}
    for name, expected in audit['rows'].items():
        p = root/'rows'/name
        if sha(p) != expected:
            raise RuntimeError('Saved result changed: '+str(p))
        rows[name] = read(p)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--scaled', type=Path)
    args = parser.parse_args()
    out = fresh('diagnostic_report')
    preflight(out, dict(stage='QUALIFICATION_REPORT_ONLY', inference_performed=False,
                        qualification=str(args.qualification), scaled=str(args.scaled)))
    raw = checked_rows(args.qualification)
    clean = [r for r in raw.values() if r.get('clean')]
    source = {str(args.qualification/'rows'/name):sha(args.qualification/'rows'/name) for name in raw}
    table = [dict(group=r['scene']['group'], method=r['method'], error_m=r['metrics']['errors_m'][0],
                  seconds=r['result']['seconds'], iterations=r['result']['counts']['iterations'],
                  stop=r['result']['stop_reason'], success20=bool(r['metrics']['success']['20'][0])) for r in clean]
    write(out/'report.json',dict(stage='PARTIAL_QUALIFICATION_NOT_FORMAL', rows=table,
                                original_inputs_verified=True, original_result_hashes_verified=True,
                                note='Source versions are identified by original contracts, not current files.'))
    csv_write(out/'clean_qualification.csv',table)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(8,4),constrained_layout=True)
    values=np.array([[next(r['error_m'] for r in table if r['method']==m and r['group']==g) for g in range(4)] for m in METHODS])
    im=ax.imshow(np.log10(np.maximum(values,.001)),cmap='viridis',aspect='auto',vmin=-3,vmax=2)
    for i in range(len(METHODS)):
        for j in range(4):
            ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]<1 else 'black')
    ax.set(xticks=range(4),xticklabels=[f'Layout {g}' for g in range(4)],yticks=range(len(METHODS)),yticklabels=METHODS)
    fig.colorbar(im,ax=ax,label='log10 error (m)')
    export(out,'01_clean_errors',fig,table,
           '初版R1无噪声孤立源盲搜核验；保留求解器加载与正则。表内单位为米，20m为判据；并非混合场景正式排名。',
           'Initial R1 noise-removed single-source qualification, with regularization retained. Cell values in metres; not a mixed-source ranking.',source,'G10-R1 / DEVELOPMENT QUALIFICATION')
    tune=[]
    fig,ax=plt.subplots(figsize=(7,3.5),constrained_layout=True)
    for i,weight in enumerate(('uniform','excess_power')):
        yy=[]
        for g in range(4):
            r=raw[f'{g}_0_0_weak_MUSIC_tune_4_{weight}.json']
            e=r['metrics']['errors_m'][0]; yy.append(e)
            tune.append(dict(group=g,segments=4,weight=weight,error_m=e))
        ax.bar(np.arange(4)+(i-.5)*.3,yy,width=.3,label=weight,color=('#0072B2','#D55E00')[i],hatch=('','//')[i])
    ax.axhline(20,ls='--',c='black',lw=.8)
    ax.set(xticks=range(4),xlabel='Fixed layout',ylabel='Single-source error (m)'); ax.legend()
    export(out,'02_music_weight',fig,tune,
           '有噪声开发孤立源，同J=4、同输入、同网格与精修，仅对比宽带汇聚权重；每布局一次实现。',
           'Noisy development singles, J=4 and identical inputs/grid/refinement; only frequency pooling differs. One realization per layout.',source,'G10-R1 / DEVELOPMENT ADAPTATION')
    if args.scaled:
        scale_rows=checked_rows(args.scaled)
        before=raw['3_0_0_weak_SBL_clean.json']; after=scale_rows['3_SBL.json']
        if before['input_sha'] != after['input_sha']:
            raise RuntimeError('Scaling diagnostic not paired')
        source[str(args.scaled/'rows/3_SBL.json')]=sha(args.scaled/'rows/3_SBL.json')
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
        dd=[]
        truth=np.array(before['truth'])[0]
        for name,r,c,ls in [('metres',before,'#0072B2','--'),('scaled coordinates',after,'#D55E00','-')]:
            h=r['result']['history']; p=np.array([v['positions'][0] for v in h]); it=[v['iteration'] for v in h]
            errors=np.linalg.norm(p-truth,axis=1); objective=[v['native_objective'] for v in h]
            axes[0].plot(it,errors,color=c,ls=ls,label=name)
            axes[1].plot(it,objective,color=c,ls=ls,label=name)
            axes[2].plot(p[:,0],p[:,1],color=c,ls=ls,label=name)
            dd.extend(dict(variant=name,iteration=n,error_m=float(e),objective=float(o),x_m=float(q[0]),y_m=float(q[1])) for n,e,o,q in zip(it,errors,objective,p))
        axes[0].axhline(20,ls=':',c='black'); axes[0].set(xlabel='SBL iteration',ylabel='Position error (m)')
        axes[1].set(xlabel='SBL iteration',ylabel='Native negative log evidence')
        axes[2].scatter(*truth,marker='*',s=90,c='black',label='True location')
        axes[2].set(xlabel='x (m)',ylabel='y (m)',aspect='equal')
        axes[0].legend(fontsize=7); axes[2].legend(fontsize=7)
        export(out,'03_sbl_scaling',fig,dd,
               '同一无噪声观测的SBL坐标数值尺度检查；字典初始间距、位置移动边界和物理目标相同。只代表该开发案例。',
               'Paired SBL coordinate-scaling diagnostic, same initial grid, physical bounds and objective. One development case only.',source,'G10-R1 / DEVELOPMENT DIAGNOSTIC')
    print('REPORT='+str(out),flush=True)


if __name__=='__main__':main()
