"""Saved result audit and conclusions for the frozen square-deployment pair."""
from itertools import permutations
import json
from pathlib import Path
import hashlib
import numpy as np
from pilot import ROOT, sha, write, observe, xia

OUT=ROOT/'square_20261027_182013_211372'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol=read(OUT/'protocol.json'); source=Path(protocol['source'])
    assert all(sha(source/name)==h for name,h in protocol['source_hashes'].items())
    checks={}; first=[]
    for path in sorted(OUT.glob('r*_G*.json')):
        r=read(path); z=dict(np.load(OUT/f'input_{r["realization"]:02d}.npz'))
        checks[path.stem+'/input']=sha(OUT/f'input_{r["realization"]:02d}.npz')==r['input_sha']
        y,_,_=observe(z,r['snr_db'],r['sir_db'],r['paths'])
        checks[path.stem+'/observation']=hashlib.sha256(y.tobytes()).hexdigest()==r['observation_sha']
        for step in r['output']['steps']:
            dec=xia.decode(np.array(step['points']),np.array(step['scores']),2,400.)
            checks[path.stem+f'/decode{step["round"]}']=dec==step['output']
        p=np.array(r['output']['steps'][0]['points']); v=np.array(r['output']['steps'][0]['scores'])
        error=float(np.linalg.norm(p[v.argmax()]-z['truth'][0])); first.append(error)
        checks[path.stem+'/fixed_target_A']=error==r['first']['initial']['A']
        if r['selected_positions'] is not None:
            pos=np.array(r['selected_positions'])
            costs=[np.sum((pos[list(order)]-z['truth'])**2,axis=1) for order in permutations([0,1])]
            best=min(costs,key=lambda q:q.sum())
            checks[path.stem+'/matching']=np.allclose(np.sqrt(best),r['evaluation']['error_m'])
    s=read(OUT/'summary.json')
    checks['aggregate_RMSE']=np.isclose(np.sqrt(np.mean(np.square(first))),s['totals']['square']['first_A_initial']['rmse_m'])
    assert all(checks.values())
    write(OUT/'post_audit.json',dict(passed=True,count=len(checks),checks={k:bool(v) for k,v in checks.items()},script_sha=sha(__file__)))
    (OUT/'square_finish.py').write_bytes(Path(__file__).read_bytes())
    conclusion='''<section id="conclusion"><h2>结论：构型有影响，但不是全部原因</h2>
<p><b>你的担忧得到部分支持。</b>对称部署让24个源位置的理想几何指标都改善，曾经得到完整位置对从25次增加到37次。不过，逐例是新增23次、丢失11次，并非原25次全部保留后再增加12次。</p>
<p><b>仍未跑通原流程。</b>最终完整位置对仍为0/84，第一轮固定A总体RMSE由2609.3m降至2480.3m，降幅约4.9%。第一峰距两源均超过100m的次数由59降至49。这个受控对照不足以把异常归结为随机部署。</p>
<p>正方形第一轮47次分组失败、37次完整；第二轮30次分组失败、6次全筛空、1次完整；最后一例在第三轮全筛空。无反射控制(G=1、SNR5、SIR3)曾有输出从2/12增加到9/12，但第一峰A RMSE反而由1989.3m变成2213.8m；有输出不能当作定位正确。</p>
<p>不应再通过换布局寻找“能跑通”的构型。建议暂时固定这个正方形控制布局，继续核对原始评分、参考频谱除法和筛选过程。本站点对照没有证明Xia原算法普遍无效，也没有确认参考频谱除法就是根因。</p>
<p class="note">本轮改动包含部署自然导致的时延、各站接收功率和接收SIR变化，不是纯GDOP单因素隔离；共同随机量和平均接收SNR定义保持不变。只有12组独立源位置/波形实现，84是配对条件数量。</p></section>'''
    page=OUT/'正方形构型对照.html'; text=page.read_text(encoding='utf-8')
    if 'id="conclusion"' not in text: text=text.replace('<section><h2>原文到底',conclusion+'<section><h2>原文到底',1)
    text=text.replace('<a href="audit.json">检查</a>','<a href="audit.json">检查</a> · <a href="post_audit.json">结果回放检查</a>')
    page.write_text(text,encoding='utf-8')
    md=f'''# 正方形部署与原随机部署配对核验

原文第2857页§V明确中心参考站(0,0)、其余四站随机部署，没有固定坐标。本轮用户授权将四个外围站等间距置于原6×6km²区域四角(±3000,±3000)m，中心站不变。此为额外构型控制，不称作者原部署复现；沿用历史对称等角四站的结构，而非历史1km最大基线绝对坐标。

复用原`pilot_20261027_170421_805496`的12组真源位置、独立BPSK、符号偏移、阴影随机量、复杂相位、相对路径增益/延迟、标准噪声与粒子种子；站点改变后按原COST231距离规律重算时延/幅度及目标平均LOS归一化。平均接收SNR定义不变，归一化噪声同一份，不是固定绝对热噪声。发射SIR不变、实际接收SIR可随几何改变。84次新求解，原算法/阈值/停止/后轮保护不变，无Hann或额外处理。

## 结果

原随机→正方形：曾有完整双源25/84→37/84；新增23次、丢失11次，共同完整子集14次；原终态完整均0/84。第一轮固定A RMSE2609.266789→2480.293836m；最近可用第一峰A RMSE2609.344799→2481.921727m，均84/84。终轮A仅77例可算，2469.450225→2549.269194m，两组缺失身份不同，不能作为完整总体配对精度结论。第一峰距两源均超过100m：59→49；100m仅诊断。

正方形第1轮47次EMPTY_SECOND_CLUSTER、37次完整；第2轮30次EMPTY_SECOND_CLUSTER、6次NO_RETAINED_PARTICLE、1次完整；第3轮1次NO_RETAINED_PARTICLE。最终与旧批都为77次第二簇缺失、7次筛空，但发生轮次/样本不同。

理想独立同方差TOA条件下，使用TDOA协方差I+11^T的几何因子sqrt(trace((J^T R^-1 J)^-1))，24个源位置全部改善，中位数3.394970→1.017015。它不是Xia/多径CRLB，不含接收SNR和局部峰效应。几何改善不保证实际算法每条件都改善。G1/SNR5/SIR3无反射组输出2→9/12，但第一轮A RMSE1989.35→2213.78m；G3/SNR20/SIR3则1718.70→1218.00m。

## 结论

构型影响确实存在，但对称部署未解决重建异常；不继续选布局求通过。建议固定此对称控制，核查式(9)原始评分/参考频谱与筛选机制。该归因下一步尚未执行。不能将本版本表现当作者算法能力上限。

## 审计

运行{s['seconds']:.3f}s，{s['audit_count']}项输入/随机量/初始粒子/身份检查，另{len(checks)}项结果回放/固定A和双源配对检查通过。输入及原评分/求解代码未变。源码与协议在运行前保存，未事后改参数。完整逐例、原输入、新输入、HTML、CSV、原始轨迹均在同目录；旧报告只增加新入口。
'''
    (OUT/'构型对照结论.md').write_text(md,encoding='utf-8')
    for target in [ROOT/'公开信息重建报告.html', ROOT/'trace_20261027_180535_576489/逐轮链路核查.html']:
        t=target.read_text(encoding='utf-8')
        relative=OUT.name+'/正方形构型对照.html' if target.parent==ROOT else '../'+OUT.name+'/正方形构型对照.html'
        if 'id="square-latest"' not in t:
            t=t.replace('<section>',f'<section id="square-latest"><h2>新增：对称正方形站点对照</h2><p>已固定五站对称布局重跑84次。几何有所改善，但原流程仍未跑通。</p><p><a href="{relative}">查看构型配对结果 →</a></p></section><section>',1)
            target.write_text(t,encoding='utf-8')
    print('Saved-result checks',len(checks),'PASS')


if __name__=='__main__': main()
