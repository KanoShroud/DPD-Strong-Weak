"""User-authorized paired change of receiver deployment only, 84 runs.

Reference stays at origin, four outer stations occupy the original area's
corners. Reuses frozen IQ generating components, shadow/phases and RNG states.
Not a claim of reproducing Xia's random-deployment Monte Carlo experiment.
"""
import csv
from collections import Counter
from datetime import datetime
import hashlib
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from pilot import ROOT, N, C, DT, sha, write, observe, evaluate, select_output, xia
from trace_audit import read, stats

BASE = ROOT/'pilot_20261027_170421_805496'
STATIONS = np.array([[0.,0.],[3000.,3000.],[-3000.,3000.],[-3000.,-3000.],[3000.,-3000.]])


def change_geometry(z, stations):
    new = {k: v.copy() for k,v in z.items()}
    old_d = np.linalg.norm(z['truth'][:,None]-z['stations'][None],axis=-1)
    d = np.linalg.norm(z['truth'][:,None]-stations[None],axis=-1)
    assert np.all(d>0)
    # Same link shadow draw and complex phase; only deterministic distance loss.
    new['stations'] = stations.copy()
    new['loss_db'] = z['loss_db']+26*np.log10(d/old_d)
    new['raw_gain'] = z['raw_gain']*(d/old_d)**(-1.3)
    new['common_scale'] = np.sqrt(np.mean(abs(new['raw_gain'][0])**2))
    new['gain'] = new['raw_gain']/new['common_scale']
    new['taps'] = z['taps']*(new['gain']/z['gain'])[:,:,None]
    tau = (d-d[:,:1])/C
    new['component'] = new['taps'][:,:,:,None]*z['spectrum'][:,None,None]*np.exp(
        -2j*np.pi*(tau[:,:,None]+DT)[...,None]*z['frequency'])
    return new


def first_metrics(result, truth):
    first = []
    for step in result['steps']:
        output=step['output']
        p=output['positions'][0] if output.get('positions') is not None else output.get('first_position')
        errors=np.linalg.norm(np.array(p)-truth,axis=1) if p is not None else None
        first.append(dict(round=step['round'],A=float(errors[0]) if errors is not None else None,
                          B=float(errors[1]) if errors is not None else None))
    valid=[r for r in first if r['A'] is not None]
    return dict(initial=first[0],terminal=first[-1],last=valid[-1] if valid else dict(A=None,B=None),steps=first)


def geometry_info(z):
    info=[]
    # Geometry-only range-difference model with equal independent TOA variances;
    # reference differencing creates covariance I+11^T. Not a Xia/MP CRLB.
    cov=np.eye(4)+np.ones((4,4))
    for truth in z['truth']:
        d=truth-z['stations']; unit=d/np.linalg.norm(d,axis=1)[:,None]
        j=unit[1:]-unit[0]
        eig=np.linalg.eigvalsh(j.T@np.linalg.solve(cov,j))
        assert np.all(eig>0)
        info.append(dict(equal_toa_geometry_factor=float(np.sqrt(np.sum(1/eig))),
                         information_condition=float(eig[-1]/eig[0])))
    return info


def main():
    started=time.perf_counter()
    frozen=read(BASE/'preflight.json')
    for p,h in frozen['sources'].items():
        assert sha(Path(p))==h and sha(BASE/Path(p).name)==h
    originals=list(BASE.glob('*.json'))+list(BASE.glob('input_*.npz'))
    before={p.name:sha(p) for p in originals}
    out=ROOT/('square_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(source=str(BASE.resolve(strict=True)),source_hashes=before,
        stations=STATIONS,approval='User requested square deployment and repeat previous pilot',
        paper_deployment='p2857 section V: center reference + four random BS; no numeric coordinates',
        square_choice='Four equidistant peripheral sites at corners of unchanged [-3000,3000]^2 region',
        historical_shape='g8/physics.py RECEIVERS: four equal-angle sites around center; shape only, not historical coordinates',
        unchanged=['truth','bits','waveforms','shadow draws','complex phases','relative taps','unit noise','particle seeds','SNR/SIR definitions','Xia solver'],
        changed=['stations','distance-related delay/loss','common normalization from target mean LOS gain'],
        noise_note='Same normalized noise samples and variance. Absolute-unit noise changes with the frozen received-SNR normalization; not fixed physical thermal noise.',
        jobs=84,source_sha=sha(__file__),scope='PAIRED_DEPLOYMENT_DIAGNOSTIC_NOT_AUTHOR_REPLICATION'))
    (out/'square_check.py').write_bytes(Path(__file__).read_bytes())
    records=[read(p) for p in sorted(BASE.glob('r*_G*.json'))]
    rows=[]; checks={}; cache={}; geometry=[]
    for idx,rec in enumerate(records):
        i=rec['realization']
        if i not in cache:
            old=dict(np.load(BASE/f'input_{i:02d}.npz'))
            identity=change_geometry(old,old['stations'])
            checks[f'{i}/identity_geometry'] = np.allclose(identity['component'],old['component'],rtol=2e-13,atol=1e-15)
            z=change_geometry(old,STATIONS)
            for k in ['truth','bits','waves','offsets','frequency','spectrum','noise_iq','noise_f']:
                checks[f'{i}/same_{k}']=np.array_equal(z[k],old[k])
            old_d=np.linalg.norm(old['truth'][:,None]-old['stations'][None],axis=-1)
            new_d=np.linalg.norm(z['truth'][:,None]-z['stations'][None],axis=-1)
            old_shadow=old['loss_db']-42.6-20*np.log10(900.)-26*np.log10(old_d/1000)
            new_shadow=z['loss_db']-42.6-20*np.log10(900.)-26*np.log10(new_d/1000)
            checks[f'{i}/same_shadow']=np.allclose(old_shadow,new_shadow,rtol=0,atol=1e-12)
            checks[f'{i}/same_relative_taps']=np.allclose(z['taps']/z['gain'][:,:,None],old['taps']/old['gain'][:,:,None],rtol=1e-13,atol=1e-15)
            checks[f'{i}/gain_loss_consistent']=np.allclose(abs(z['raw_gain']),10**(-z['loss_db']/20),rtol=1e-13,atol=0)
            checks[f'{i}/snr_normalization']=np.isclose(np.mean(abs(z['gain'][0])**2),1)
            assert all(checks.values())
            np.savez_compressed(out/f'input_{i:02d}.npz',**z)
            cache[i]=(old,z)
            geometry.append(dict(realization=i,truth=z['truth'],old_stations=old['stations'],square_stations=STATIONS,
                                 old_geometry=geometry_info(old),square_geometry=geometry_info(z)))
        old,z=cache[i]
        y,components,sigma=observe(z,rec['snr_db'],rec['sir_db'],rec['paths'])
        old_y,old_components,_=observe(old,rec['snr_db'],rec['sir_db'],rec['paths'])
        checks[f'{idx}/old_observation_sha']=hashlib.sha256(old_y.tobytes()).hexdigest()==rec['observation_sha']
        checks[f'{idx}/same_normalized_noise']=np.allclose(y-components.sum(0),old_y-old_components.sum(0),rtol=1e-12,atol=1e-15)
        assert all(checks.values())
        result=xia.solve(y,z['frequency'],z['stations'],2,N,3000,seed=rec['particle_seed'])
        checks[f'{idx}/same_initial_particles']=result['steps'][0]['points']==rec['output']['steps'][0]['points']
        pred,round_id,retained=select_output(result)
        row=dict(realization=i,snr_db=rec['snr_db'],sir_db=rec['sir_db'],paths=rec['paths'],
                 old_evaluation=rec['evaluation'],evaluation=evaluate(pred,z['truth']),
                 old_first=first_metrics(rec['output'],old['truth']),first=first_metrics(result,z['truth']),
                 old_terminal=rec['output']['status'],terminal=result['status'],
                 terminal_complete=result['positions'] is not None,selected_round=round_id,retained_previous=retained,
                 actual_los_received_sir_db=float(10*np.log10(np.mean(abs(z['gain'][0])**2)/(np.mean(abs(z['gain'][1])**2)*10**(-rec['sir_db']/10)))),
                 old_actual_los_received_sir_db=rec['actual_los_received_sir_db'],seconds=result['seconds'])
        name=f'r{i:02d}_snr{rec["snr_db"]:g}_sir{rec["sir_db"]:g}_G{rec["paths"]}'
        write(out/(name+'.json'),dict(**row,output=result,selected_positions=pred,
            original_record_sha=sha(BASE/(name+'.json')),input_sha=sha(out/f'input_{i:02d}.npz'),
            observation_sha=hashlib.sha256(y.tobytes()).hexdigest()))
        rows.append(row)
        print(f'正方形配对 {idx+1}/84 | {name} | {result["status"]}',flush=True)
    for name,h in before.items(): checks['source/'+name]=sha(BASE/name)==h
    for p,h in frozen['sources'].items(): checks['solver/'+p]=sha(Path(p))==h
    assert all(checks.values())
    write(out/'geometry.json',geometry)
    write(out/'result.json',dict(rows=rows,seconds=time.perf_counter()-started,checks={k:bool(v) for k,v in checks.items()}))
    write(out/'audit.json',dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()}))
    report(out,rows,geometry,time.perf_counter()-started,len(checks))
    print('REPORT',out,flush=True)


def report(out,rows,geometry,seconds,check_count):
    groups=[]
    for g,snr,sir in sorted(set((r['paths'],r['snr_db'],r['sir_db']) for r in rows)):
        sub=[r for r in rows if (r['paths'],r['snr_db'],r['sir_db'])==(g,snr,sir)]
        for version in ['old','square']:
            ek='old_evaluation' if version=='old' else 'evaluation'
            fk='old_first' if version=='old' else 'first'
            errors=[r[ek]['error_m'] for r in sub if r[ek]['complete']]
            groups.append(dict(G=g,SNR=snr,SIR=sir,version=version,n=len(sub),
                complete=sum(r[ek]['complete'] for r in sub),both20=sum(r[ek]['both20'] for r in sub),
                first_A_initial=stats([r[fk]['initial']['A'] for r in sub]),
                first_A_last=stats([r[fk]['last']['A'] for r in sub]),
                first_A_terminal=stats([r[fk]['terminal']['A'] for r in sub]),
                matched_A_conditional=stats([e[0] for e in errors]),matched_B_conditional=stats([e[1] for e in errors])))
    totals={}
    for version in ['old','square']:
        fk='old_first' if version=='old' else 'first'; ek='old_evaluation' if version=='old' else 'evaluation'
        totals[version]=dict(complete=sum(r[ek]['complete'] for r in rows),both20=sum(r[ek]['both20'] for r in rows),
            first_A_initial=stats([r[fk]['initial']['A'] for r in rows]),
            first_A_last=stats([r[fk]['last']['A'] for r in rows]),
            first_A_terminal=stats([r[fk]['terminal']['A'] for r in rows]),
            first_far_from_both100=sum(min(r[fk]['initial']['A'],r[fk]['initial']['B'])>100 for r in rows if r[fk]['initial']['A'] is not None),
            terminal_status=dict(Counter(r['old_terminal' if version=='old' else 'terminal'] for r in rows)))
    trajectory={}
    for j in [1,2,3]:
        recs=[read(p) for p in out.glob('r*_G*.json')]
        states=[s['output']['status'] for r in recs for s in r['output']['steps'] if s['round']==j]
        trajectory[str(j)]=dict(Counter(states))
    old_geo=np.array([[v['equal_toa_geometry_factor'] for v in r['old_geometry']] for r in geometry])
    new_geo=np.array([[v['equal_toa_geometry_factor'] for v in r['square_geometry']] for r in geometry])
    summary=dict(totals=totals,groups=groups,seconds=seconds,audit_count=check_count,square_rounds=trajectory,
        geometry=dict(old_median=float(np.median(old_geo)),square_median=float(np.median(new_geo)),
                      improved=int(np.sum(new_geo<old_geo)),total=24,
                      note='Equal-independent-TOA geometry factor only; not Xia accuracy prediction or multipath CRLB'))
    write(out/'summary.json',summary)
    flat=[dict(G=r['G'],SNR=r['SNR'],SIR=r['SIR'],version=r['version'],n=r['n'],complete=r['complete'],both20=r['both20'],
               first_A_initial_RMSE=r['first_A_initial']['rmse_m'],first_A_last_RMSE=r['first_A_last']['rmse_m'],
               matched_A_conditional_RMSE=r['matched_A_conditional']['rmse_m'],matched_B_conditional_RMSE=r['matched_B_conditional']['rmse_m']) for r in groups]
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    # Same-input complete subset, so pair RMSE comparisons do not hide missingness.
    common=[r for r in rows if r['evaluation']['complete'] and r['old_evaluation']['complete']]
    write(out/'common_complete.json',dict(n=len(common),old_A=stats([r['old_evaluation']['error_m'][0] for r in common]),
        square_A=stats([r['evaluation']['error_m'][0] for r in common]),
        old_B=stats([r['old_evaluation']['error_m'][1] for r in common]),square_B=stats([r['evaluation']['error_m'][1] for r in common])))
    table=''.join(f'<tr><td>{r["G"]}</td><td>{r["SNR"]:g}</td><td>{r["SIR"]:+g}</td><td>{"随机" if r["version"]=="old" else "正方形"}</td><td>{r["complete"]}/12</td><td>{r["first_A_initial"]["rmse_m"]:.1f}</td><td>{r["first_A_last"]["rmse_m"]:.1f}</td></tr>' for r in groups)
    t=totals
    main_table=''.join(f'<tr><td>{name}</td><td>{t[v]["complete"]}/84</td><td>{t[v]["first_A_initial"]["rmse_m"]:.1f}</td><td>{t[v]["first_A_last"]["rmse_m"]:.1f}</td><td>{t[v]["first_far_from_both100"]}/84</td></tr>' for v,name in [('old','原随机部署'),('square','固定正方形')])
    # Plain SVG; all sources included, not selected for good/bad outcomes.
    svg='<svg viewBox="0 0 470 450" role="img" aria-label="固定正方形五站与12组真实源"><rect x="45" y="35" width="360" height="360" fill="#f5f8f9" stroke="#bacace"/>'
    for point in STATIONS:
        x=45+(point[0]+3000)/6000*360;y=395-(point[1]+3000)/6000*360
        svg+=f'<rect x="{x-5}" y="{y-5}" width="10" height="10" fill="#234d66"/>'
    for g in geometry:
        for q,p in enumerate(g['truth']):
            x=45+(p[0]+3000)/6000*360;y=395-(p[1]+3000)/6000*360
            svg+=f'<circle cx="{x}" cy="{y}" r="4" fill="{["#087e70","#bd6932"][q]}"/><text x="{x+5}" y="{y-5}" font-size="10">{g["realization"]}{["A","B"][q]}</text>'
    svg+='<text x="45" y="425" font-size="13">横纵范围均为−3000～3000 m；蓝方块为站点</text></svg>'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>正方形站点配对核验</title><style>body{{max-width:1060px;margin:auto;padding:32px 22px;background:#f4f4ee;color:#203c47;font:16px/1.85 "Microsoft YaHei",system-ui}}section{{background:white;padding:24px;border:1px solid #dce5e6;border-radius:14px;margin:20px 0}}h1{{font-size:30px}}h2{{font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #dce5e6}}th{{background:#edf5f3}}.scroll{{overflow:auto}}.note{{padding:14px;background:#fff1d8}}svg{{width:100%;max-width:470px}}a{{color:#007b72}}summary{{cursor:pointer}}</style>
<p>2026-10-27 · 只替换部署 · 原84次条件逐例配对</p><h1>换成对称正方形后，Xia重建能否跑通？</h1>
<section><h2>原文到底怎样部署？</h2><p>原文第2857页第V节规定：区域6×6km²，参考站位于中心(0,0)，其余4站随机部署；没有固定坐标表。因此原来的随机部署不是擅自加入的场景。本次依用户要求做额外的对称布局控制，不称为原随机部署实验的严格复现。</p><p>固定5站：(0,0)、(3000,3000)、(−3000,3000)、(−3000,−3000)、(3000,−3000)m。沿用历史等角四站的结构，外围四站相邻间距6000m；不是直接复制历史1000m最大基线坐标。</p>{svg}</section>
<section><h2>哪些变了，哪些没变？</h2><p>只改站坐标，并按原距离规律重算时延和衰减；12组源坐标、波形、阴影随机量、所有相位、相对反射幅度与时延、标准噪声、初始1000粒子及算法参数保持不变。</p><p>继续按目标LOS五站平均接收SNR设置噪声。公共数值归一化随新距离重算；归一化后的噪声样本和方差完全相同，绝对物理单位下的噪声并非固定。这不是只改TDOA、冻结接收功率的实验：各站功率与接收SIR随部署自然变化。</p></section>
<section><h2>配对结果</h2><div class="scroll"><table><tr><th>部署</th><th>曾有完整双源输出</th><th>第一轮固定A RMSE/m</th><th>最近可用第一峰A RMSE/m</th><th>第一峰离两源均&gt;100m</th></tr>{main_table}</table></div><p class="note">有输出不等于定位成功。第一峰固定配给A，不按真值换身份；100m仅是诊断尺度。完整位置对另按原匹配规则评价，CSV同时保留分母。总体数字混合7个条件，分条件表用于检查是否由少数条件主导。</p><details><summary>七个条件的全部结果</summary><div class="scroll"><table><tr><th>路径数G</th><th>SNR</th><th>SIR</th><th>布局</th><th>完整输出</th><th>第一轮A RMSE/m</th><th>最近可用A RMSE/m</th></tr>{table}</table></div></details><p>正方形终态：{t['square']['terminal_status']}；各轮统计见summary.json。没有因结果不好修改阈值、噪声或源位置。</p></section>
<section><h2>几何本身是否改善？</h2><p>只看理想等方差TOA所推导的几何指标，24个源位置中有{np.sum(new_geo<old_geo)}/24个改善，中位数由{np.median(old_geo):.3f}降至{np.median(new_geo):.3f}。此指标考虑共享参考站产生的TDOA噪声相关性。</p><p>这只是局部几何诊断，不包含多径、接收功率差、Xia评分或搜索过程，不能拿来代替算法RMSE或称为实际CRLB。正方形也不保证每个目标位置都最好。</p></section>
<section><h2>复现与审计</h2><p>新增84次求解，运行及检查共{seconds:.1f}s，{check_count}项检查通过。原输入与算法文件身份保持不变，逐例验证初始粒子完全相同。</p><p><a href="protocol.json">冻结协议</a> · <a href="audit.json">检查</a> · <a href="summary.json">汇总</a> · <a href="summary.csv">CSV</a> · <a href="common_complete.json">共同有输出子集</a> · <a href="geometry.json">几何</a> · <a href="result.json">逐例</a> · <a href="square_check.py">代码</a></p></section></html>'''
    (out/'正方形构型对照.html').write_text(page,encoding='utf-8')
    print(summary)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
