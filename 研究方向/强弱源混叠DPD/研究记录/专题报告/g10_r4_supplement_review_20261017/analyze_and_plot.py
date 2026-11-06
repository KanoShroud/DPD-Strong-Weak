"""Read frozen results; reconstruct two diagnostic spectra on one CPU thread only."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import psutil
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUN=ROOT/'outputs_e2e/unified/e2e_g10/r4/formal_recovered_20261017_093236_136920'
STYLE=Path('C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle')
STATIONS=np.stack([500*np.cos(np.arange(4)*np.pi/2),500*np.sin(np.arange(4)*np.pi/2)],axis=1).astype(np.float32).astype(float)
MANIFEST={}

def payload(path):
    b=path.read_bytes(); MANIFEST[str(path)]=hashlib.sha256(b).hexdigest(); return b

def read(path):
    return json.loads(payload(path))

def csv_out(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def response(points,f):
    d=np.maximum(np.linalg.norm(np.asarray(points)[:,None]-STATIONS,axis=-1),1.)
    a=np.exp(-2j*np.pi*f[:,None,None]*d.T[None]/299792458.)/d.T[None]
    return a/np.linalg.norm(a,axis=1,keepdims=True)

def case(rep):
    prefix=f'3_{rep}_0'; r=read(RUN/'rows'/(prefix+'_AP_5_JOINT.json'))
    meta=read(RUN/'inputs'/(prefix+'.json'))
    b=payload(RUN/meta['path']);assert hashlib.sha256(b).hexdigest()==meta['sha256']
    with np.load(io.BytesIO(b),allow_pickle=False) as z:
        iq=z['signal'].sum(0)+z['noise']; truth=z['positions'].copy()
    assert hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()==r['input_identity']['iq_sha']
    assert hashlib.sha256(payload(RUN/'inputs'/(prefix+'.json'))).hexdigest()==r['input_identity']['metadata_sha']
    n=iq.shape[1]; assert n==8192
    f=np.fft.fftfreq(n,1/50e6); keep=abs(f)<=7e6
    x=np.fft.fft(iq).T[keep]/n; f=f[keep];energy=float(np.sum(abs(x)**2))
    mech=r['result']['mechanism']; first=np.array(mech['first_position'])
    a0=response(first,f)[:,:,0]
    residual=x-a0*np.sum(a0.conj()*x,axis=1)[:,None]
    base=float(np.sum(abs(residual)**2)/energy)
    axis=np.arange(-1000.,1000.1,20.);xx,yy=np.meshgrid(axis,axis)
    points=np.stack([xx.ravel(),yy.ravel()],axis=1); unconditioned=[]; conditional=[]
    start=time.perf_counter()
    cache=OUT/(prefix+'_spectra.csv')
    if cache.exists():
        saved=list(csv.DictReader(io.StringIO(payload(cache).decode('utf-8-sig'))))
        np.testing.assert_array_equal([[float(v['x_m']),float(v['y_m'])] for v in saved],points)
        unconditioned=[float(v['single_source_ls']) for v in saved]
        conditional=[float(v['conditional_score']) for v in saved]
    else:
        for lo in range(0,len(points),32):
            a=response(points[lo:lo+32],f)
            unconditioned.extend(np.sum(abs(np.sum(a.conj()*x[:,:,None],axis=1))**2,axis=0)/energy)
            u=a-a0[:,:,None]*np.sum(a0.conj()[:,:,None]*a,axis=1)[:,None,:]
            norm=np.sum(abs(u)**2,axis=1)
            dot=np.sum(u.conj()*residual[:,:,None],axis=1)
            conditional.extend(np.sum(np.where(norm>1e-20,abs(dot)**2/np.maximum(norm,1e-20),0),axis=0)/energy)
    conditional=np.array(conditional); unconditioned=np.array(unconditioned)
    ids=np.flatnonzero(conditional.reshape(101,101)==maximum_filter(conditional.reshape(101,101),3,mode='constant',cval=-np.inf))
    chosen=[]
    for i in ids[np.argsort(-conditional[ids],kind='stable')]:
        if all(np.linalg.norm(points[i]-points[j])>=30 for j in chosen):chosen.append(i)
        if len(chosen)==5:break
    np.testing.assert_array_equal(points[chosen],np.array(mech['candidate_list'])[:,1])
    errors=[]
    for i,branch in zip(chosen,mech['branches']):
        errors.append(abs((base-conditional[i])-branch['initial_objective']))
    assert max(errors)<1e-10, errors
    csv_out(OUT/(prefix+'_spectra.csv'),[dict(x_m=p[0],y_m=p[1],single_source_ls=s,conditional_score=q) for p,s,q in zip(points,unconditioned,conditional)])
    csv_out(OUT/(prefix+'_branches.csv'),[dict(rank=b['rank'],initial_x=b['initial_positions'][1][0],initial_y=b['initial_positions'][1][1],final_x=b['positions'][1][0],final_y=b['positions'][1][1],initial_objective=b['initial_objective'],final_objective=b['objective']) for b in mech['branches']])
    print(prefix,'CPU谱重建通过；候选顺序一致，初始目标误差',max(errors),'用时',round(time.perf_counter()-start,2),'s',flush=True)
    return dict(prefix=prefix,rep=rep,row=r,truth=truth,mech=mech,points=points,
        single=unconditioned,conditional=conditional,max_initial_objective_error=max(errors))

def axes_base(ax):
    ax.set(xlim=(-1000,1000),ylim=(-1000,1000),xlabel='x (m)',ylabel='y (m)')
    ax.set_aspect('equal');ax.set_xticks([-1000,-500,0,500,1000]);ax.set_yticks([-1000,-500,0,500,1000])

def markers(ax,c,candidates=True,heat=False):
    ax.scatter(*STATIONS.T,marker='^',s=44,c='white' if heat else '#444444',edgecolor='black',label='接收站',zorder=5)
    ax.scatter(*c['truth'][0],marker='*',s=150,c='#56B4E9',edgecolor='black',label='源A真值',zorder=7)
    ax.scatter(*c['truth'][1],marker='*',s=150,c='#E69F00',edgecolor='black',label='源B真值',zorder=7)
    if candidates:
        for b in c['mech']['branches']:
            p=np.array(b['initial_positions'])[1]
            ax.scatter(*p,marker='D',s=37,facecolor='white',edgecolor='black',zorder=8,clip_on=False,label='粗候选' if b['rank']==1 else None)
            ax.annotate(str(b['rank']),p,xytext=(6,4),textcoords='offset points',fontsize=10,color='black',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.4),zorder=9)

def save(fig,name,caption):
    for suffix in ('pdf','png'):fig.savefig(OUT/(name+'.'+suffix),dpi=600,bbox_inches='tight')
    plt.close(fig)
    (OUT/(name+'_说明.md')).write_text(caption,encoding='utf-8')

def plots(cases):
    plt.style.use(STYLE)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':9})
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,c in zip(axes,cases):
        axes_base(ax);markers(ax,c)
        hull=np.vstack([STATIONS,STATIONS[0]])
        ax.plot(*hull.T,'--',color='.55',lw=.8)
        for b in c['mech']['branches']:
            ax.annotate('',xy=b['positions'][1],xytext=b['initial_positions'][1],arrowprops=dict(arrowstyle='->',lw=1,color='#D55E00',alpha=.7))
            ax.scatter(*b['positions'][1],marker='x',s=42,c='#D55E00',zorder=6,label='精修终点' if b['rank']==1 else None)
        ax.set_title(f'实现{c["rep"]}：'+('恢复成功' if all(c['row']['metrics']['success']['20']) else '五候选仍失败'))
    axes[0].legend(loc='upper left');fig.suptitle('同一源—站构型，不同波形/噪声实现；0 dB等功率')
    save(fig,'geometry_candidates','源—站位置、原始粗候选排名及精修终点。箭头仅连接保存的起点与终点，不表示真实迭代路径。两个源等平均接收功率，A/B为生成身份。坐标单位m。\nGeometry, ranked candidates and refinement endpoints. Arrows connect endpoints, not optimizer trajectories. Equal-power sources.\n')
    for kind,title,floor in [('single','未投影的单源LS空间评分',-10),('conditional','固定第一源估计后的条件投影空间评分',-1)]:
        fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
        for ax,c in zip(axes,cases):
            values=c[kind]; db=10*np.log10(np.maximum(values/values.max(),1e-14))
            im=ax.imshow(db.reshape(101,101),origin='lower',extent=(-1010,1010,-1010,1010),vmin=floor,vmax=0,cmap='viridis',interpolation='nearest')
            axes_base(ax);markers(ax,c,candidates=kind=='conditional',heat=True)
            ax.set_title(f'实现{c["rep"]}；'+('恢复成功' if all(c['row']['metrics']['success']['20']) else '未恢复'))
        fig.colorbar(im,ax=axes,shrink=.8,label='相对本图最大值 (dB)')
        axes[0].legend(loc='upper left',frameon=True,facecolor='white',framealpha=.95);fig.suptitle(title+'（20 m网格）')
        save(fig,kind+'_spectra',f'基于原始保存IQ在CPU上重建，同一50MHz/8192点、±7MHz、20m网格与原评分公式。每图除以自身最大值转dB，低于{floor}dB仅在显示颜色中截断，原始CSV未截断。条件图突出最强1dB范围，不与单源图的颜色亮度直接比较。无定位重跑、无GPU调用。\nCPU reconstruction of saved IQ using the unchanged score. Each panel normalized to its own maximum; display floor only, raw CSV retained. Not a new localization experiment.\n')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    for ax,c in zip(axes,cases):
        bb=c['mech']['branches'];rank=[b['rank'] for b in bb]
        ax.plot(rank,[b['initial_objective'] for b in bb],'o--',c='#0072B2',label='粗候选初始目标')
        ax.plot(rank,[b['objective'] for b in bb],'s-',c='#D55E00',label='联合精修后目标')
        ax.set(xlabel='候选排名',ylabel='归一化联合残差（越小越好）',xticks=rank,title=f'实现{c["rep"]}')
        ax.legend();ax.grid(alpha=.15)
    save(fig,'branch_objectives','从已保存的分支目标直接绘制；初始排序并不保证精修后的排序。\nSaved branch objectives before and after joint refinement.\n')

def main():
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    audit=read(RUN/'supplement_audit_report.json')
    for name,h in audit['rows'].items():
        if name.endswith(('_BNOMP_REF.json','_PROFILE_DIAG.json')):
            assert hashlib.sha256(payload(RUN/'rows'/name)).hexdigest()==h
    assert audit['passed'] and audit['solves']==16320
    read(RUN/'reports/supplement/comparison_report.json')
    cases=[case(400),case(405)];plots(cases)
    for c in cases:
        name=c['prefix']+'_AP_5_JOINT.json'; assert MANIFEST[str(RUN/'rows'/name)]==audit['rows'][name]
    for path,h in MANIFEST.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    result=dict(passed=True,gpu_used=False,cpu_threads=1,reconstructed_spectra_only=True,
        no_optimizer_called=True,source_files=MANIFEST,case_checks={c['prefix']:c['max_initial_objective_error'] for c in cases},
        plot_code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('绘图完成：'+str(OUT),flush=True)

if __name__=='__main__':main()
