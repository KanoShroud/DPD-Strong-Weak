"""Saved-evidence atlas. No optimizer imports, simulation, or result replacement."""
from pathlib import Path
import hashlib
import html
import json
import math
from urllib.parse import quote

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CAL = BASE/'risk_calibration_20261102/formal_20261102_181736_622970'
REPLAY = BASE/'risk_replay_20261102/formal_20261102_223046_550434'
XIA = BASE/'xia_scale_20261103/run_20261103_180022_034778'
PAIR = BASE/'strong_neighbor_20261103/paired/run_20261103_145645_539600'
BOOT = BASE/'risk_bootstrap_20261103/run_20261103_191510_578176'
DIAG = BASE/'failure_attribution_20261104/run_20261104_002300_517861'
PREC = DIAG/'precision_003759_457922'
COND = ('L0', 'L2', 'L3', 'HIGH')
SOURCES = {}
CHECKS = {}


def mark(p):
    p = Path(p).resolve(strict=True)
    assert p.is_relative_to(BASE)
    SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(p):
    d = json.loads(mark(p).read_text(encoding='utf-8'))
    return d.get('payload', d) if isinstance(d, dict) else d


def esc(x):
    return html.escape(str(x))


def link(p, label):
    p = mark(p)
    rel = '../'+p.relative_to(BASE).as_posix()
    return f'<a href="{quote(rel, safe="/")}">{esc(label)}</a>'


def table(head, rows, caption=''):
    return '<div class="table-wrap"><table>'+ ('<caption>'+caption+'</caption>' if caption else '')+'<thead><tr>'+''.join('<th scope="col">'+str(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def fmt(v, digits=3):
    return '—' if v is None else f'{v:,.{digits}f}'


def rmse(p, truth):
    p = np.asarray(p)
    return math.sqrt(min(np.sum((p-truth)**2), np.sum((p[::-1]-truth)**2))/2)


def section(num, title, content, ident):
    return f'<section id="{ident}"><div class="eyebrow">{num}</div><h2>{title}</h2>{content}</section>'


contract = read(CAL/'contract.json')['contract']
xia, pair = read(XIA/'summary.json'), read(PAIR/'summary.json')
replay, boot = read(REPLAY/'summary.json'), read(BOOT/'final_analysis.json')
diag, precision = read(DIAG/'result.json'), read(PREC/'result.json')
cal_summary = read(CAL/'summary.json')
bank, solved, metrics, records = {}, {}, {}, []
for cond in COND:
    p = mark(contract['conditions'][cond]['bank'])
    with np.load(p, allow_pickle=False) as z:
        bank[cond] = {k: z[k] for k in ('p', 'gains', 'delays', 'weights', 'los_snr_db', 'variance')}
    losses = {k: [] for k in ('GEO', 'LOS', 'WIDE')}
    for rep in range(8):
        ident = f'{cond}_R{rep:02d}'
        s = read(CAL/'cases'/ident/'solve.json')
        solved[ident] = s
        truth = bank[cond]['p']
        row = {'case': ident}
        for method, pos in [('GEO', s['initial']), ('LOS', s['los']['p']), ('WIDE', s['final']['p'])]:
            value = rmse(pos, truth)
            row[method] = value
            losses[method].append(value**2)
        row.update(K=len(s['paths']), stop=s['stop'], seconds=s['seconds'],
                   LOS_seconds=s['frontend_seconds']+s['los']['seconds'],
                   los_pg=s['los']['projected_gradient'], final_pg=s['final']['projected_gradient'])
        records.append(row)
    metrics[cond] = {k: math.sqrt(np.mean(v)) for k, v in losses.items()}
    for k in losses:
        CHECKS[f'{cond}/{k}/raw_matches_saved_summary'] = bool(np.isclose(metrics[cond][k], xia['metrics'][cond][k]['full_rmse_m'], rtol=1e-11))
CHECKS['32_records_no_drops'] = len(records) == 32
CHECKS['one_geometry'] = all(np.array_equal(bank[c]['p'], bank['L0']['p']) for c in COND)
CHECKS['precision_rankings_stable'] = precision['all_rankings_unchanged']
CHECKS['selected_precision_8_pass'] = all(r['first' if r['selected'] == 1 else 'second']['accepted'] for r in precision['rows'])
CHECKS['all_precision_14_of16_reported_not16'] = sum(r[k]['accepted'] for r in precision['rows'] for k in ('first','second')) == 14
for c in diag['comparisons']:
    s = read(DIAG/(c['case']+'_alternative.json'))
    CHECKS[c['case']+'/alternative_rmse'] = bool(np.isclose(rmse(s['final']['p'], bank['HIGH']['p']), c['alternative_rmse_m'], rtol=1e-12))

# Model-independent rendering of the actual geometry, not an imagined environment.
stations = [(0,0), (3000,3000), (-3000,3000), (-3000,-3000), (3000,-3000)]
svg = '<svg viewBox="0 0 680 460" role="img" aria-label="实际五站和双源二维构型，单位米">'
svg += '<rect x="120" y="35" width="390" height="390" rx="8" fill="#edf5f5" stroke="#9ab9bc"/>'
for v in (-3000,0,3000):
    x, y = 315+v*.065, 230-v*.065
    svg += f'<path d="M{x},35V425 M120,{y}H510" stroke="#cbdada" stroke-dasharray="4 5"/><text x="{x}" y="448" text-anchor="middle">{v}</text>'
for i,(x,y) in enumerate(stations):
    sx,sy=315+x*.065,230-y*.065
    svg += f'<rect x="{sx-6}" y="{sy-6}" width="12" height="12" rx="2" fill="#164b5b"/><text x="{sx+12}" y="{sy+5}">站{i}</text>'
for i,(x,y) in enumerate(bank['L0']['p']):
    sx,sy=315+x*.065,230-y*.065
    svg += f'<circle cx="{sx}" cy="{sy}" r="8" fill="#d77843"/><text x="{sx-12}" y="{sy-15}" text-anchor="end">源{chr(65+i)}</text>'
svg += '<text x="535" y="445">x / m</text><text x="70" y="30">y / m</text></svg>'

params = [
 ('空间与运动','二维静态，5站、2源；无墙面/反射体地图，不估计速度/多普勒','位置未知；站坐标、源数2已知'),
 ('站点配置','中心1站＋正方形四角；x、y各在±3000 m；见坐标表','所有方法同样知道5站坐标与搜索域'),
 ('数据规模','L20一个源站构型；L0/L2/L3/HIGH各8次噪声重复，共32条','不是32个独立布局；不是每次重新生成波形/信道'),
 ('采样与记录','50 MHz；8192点；163.84 μs；FFT间隔约6.104 kHz','同步、公共采样轴；没有同步误差实验'),
 ('信号','两个独立BPSK数据源；10 MBd；RRC滚降0.25；总带宽12.5 MHz','不向估计器提供比特、脉冲模板、源波形'),
 ('波形生成','非周期连续时间RRC延时合成后取有限记录；每采样点合成64个邻近符号项（SPAN=32）','没有循环平移代替物理时延'),
 ('输入预处理','不加Hann；估计使用±6.25 MHz；8192点正交归一FFT保留2049频点','带外频点用于估计噪声尺度，不参与定位目标'),
 ('直达增益','幅度1000/源站距离，链路独立随机相位；单条记录内固定','估计器拟合未知跨站复增益，不获真实增益'),
 ('源间强弱','没有扫描两源功率差；按当前距离与波形自然产生各站功率差','HIGH指反射总功率高，不指强弱源功率差'),
 ('噪声与SNR','独立复高斯白噪声；以源B直达功率在五站的平均带内SNR=6 dB定标','不是各站各源都6 dB；生成方差不交给求解器'),
 ('路径关联','同源所有路径共享同一未知波形；两源波形独立','每条路径均为所属源的延迟副本，不是独立噪声'),
 ('路径数','各链路真实反射数0 / 2 / 3 / 4；全系统反射数0 / 20 / 30 / 40','不告诉估计器真实数目；64为公共总容量保护'),
 ('允许的功率先验','每条反射功率≤本链路直达功率；不限制反射功率之和','HIGH允许总反射功率超过直达'),
 ('生成时延','BΔτ来自[3,3.5]、[7,7.5]、[11,11.5]、[15,15.5]四组','这些真实区间和最小间隔不交给算法'),
 ('算法时延范围','增径枚举1…8191个采样延时；连续优化额外时延0…163.84 μs','搜索尺度来自记录长度，不用真实路径支持范围'),
 ('正式重复种子','SeedSequence([2026110203, 条件序号0…3, 重复序号0…7])','L0/L2/HIGH共享原信号银行；L3使用另一波形/信道银行'),
]
coordinate_rows = [(f'站{i}',x,y,'已知') for i,(x,y) in enumerate(stations)]
coordinate_rows += [(f'源{chr(65+i)}',fmt(x,6),fmt(y,6),'仅生成与评价知道') for i,(x,y) in enumerate(bank['L0']['p'])]
channel_rows=[]
for cond,k,ids in [('L0',0,[]),('L2',2,[0,3]),('L3',3,[0,1,3]),('HIGH',4,[0,1,2,3])]:
    ratio=0 if k==0 else 1.2 if cond=='HIGH' else 10**(-.6)
    if k:
        w=bank[cond]['weights'][:,:,ids]
        powers=ratio*w/w.sum(axis=2,keepdims=True)
        intervals='；'.join(f'{a/B*1e6:.2f}–{(a+.5)/B*1e6:.2f}' for a in [3,7,11,15] if [3,7,11,15].index(a) in ids) if (B:=12.5e6) else ''
        pr=f'{powers.min():.3f}–{powers.max():.3f}'
    else:
        intervals,pr='无','0'
    channel_rows.append((cond,f'{k} / {10*k}',f'{ratio:.4f}', '∞' if not k else f'{-10*math.log10(ratio):.3f}',pr,intervals))

methods=[
 ('GEO 几何初始化','10站对互相关模；每对取最多6个局部峰→非共线三站解析候选→五站截断残差精修→联合选两个位置','不拟合原IQ多径；以所有60个峰的覆盖奖励选位置对','32条；原流程前端阶段'),
 ('LOS-DPD','由GEO位置开始；秩一初估跨站复增益；每频点LS消去两源波形；联合精修4个位置坐标和16个实增益参数','反射数固定0；原IQ拟合；有界least_squares/TRF，最多300次函数评价','32条；同一流程的LOS阶段'),
 ('未知阶数多径DPD / WIDE','同一GEO/LOS起点；逐轮搜索源—站—额外延时；取通过检验的最强增量；加一分量后联合精修所有位置/增益/时延','未知波形VP＋SLSQP；单条反射圆盘约束；数值收尾；按增径检验停止，最多64个总分量','32条完整输出；30条正常停止，2条容量触顶'),
 ('插件位置风险选模','先完成上述整条增径轨迹；以完整模型作参照，对可评分状态计算位置局部敏感性和插件风险；取最小分数，平局低阶','不重新找位置、不删除重拟合路径；完整参照不适用时回退原输出','32条完整回放；不是额外32条新数据'),
 ('完整重采样 / bootstrap','用L0/L2/L3各R00的拟合信号作生成参照，各加16次噪声；每次重新盲初始化和完整增径','对最多接受k次增径的cap政策估风险；冻结所选cap后评价原R01–R07','48次拟合世界重定位；21条旧接收记录迁移评价'),
 ('Flood固定K运输LP适配','同观测提峰与三站候选；用运输LP关联时差和位置候选；选2源；关联时差上非线性精修','K=2；代价为距离差残差²/(c/B)²；离群代价1，激活代价1，容量10，关联权重>0.7；精修最多100次','32条；两步定位，不是IQ多径DPD；不是我方后端'),
 ('Xia公共候选适配（历史）','三站几何候选替换原随机首轮池，之后用Xia评分、筛选和聚类','首轮分布已改变；有单列结果，不能冒称独立原生流程','32条；保留历史但不作为最新版Xia'),
 ('Xia独立随机流程／公开尺度适配','全搜索域随机粒子；参考站归一跨谱乘积评分；10箱阈值；400 m聚类；10 m高斯重采样；最多3轮','每轮4,703,331粒子；未提供真值；8阶Taylor/FFT数值核保持原评分；终轮缺失不补零','32条；独立前端；公开区域尺度适配，不是原论文参数原封不动复现'),
 ('两区域LOS筛选（本轮诊断）','原几何最优对为第一支；跳过距第一对≤c/B的近重复对，取下一最高奖励的不同位置对；两支分别做LOS，选较低J','只诊断两个已有位置区域；不是LOS全域搜索；不使用位置真值挑分支','HIGH 8条LOS排序；仅2条错误原分支另做完整多径重算'),
 ('固定清单局部响应','用观测R00确定路径归属/条数和共同局部起点；R01–R07仅重估连续参数；另做小扰动','不是真路径oracle；不同于完整盲流程；未实施严格同起点“重新选支持”对照','14次条件重拟合＋8次扰动＋2次donor拟合'),
 ('EM/MM数学桥接','共同波形E/M代理、梯度与数值上界核验','只有数学内核；未完成独立前端/未知阶数完整流程','27项检查；没有可排名的完整定位成绩'),
]

results=[]
for cond in COND:
    cells=[cond]+[fmt(metrics[cond][m],6) for m in ('GEO','LOS','WIDE')]
    cells.append(fmt(pair['metrics'][cond]['Flood']['full_rmse_m'],6))
    x=xia['metrics'][cond]['Xia_scale_terminal']
    cells.append(fmt(x['full_rmse_m'],6) if x['full_rmse_m'] is not None else f'全体未定义<br><small>完整{x["complete"]}/8条件值 {fmt(x["conditional_rmse_m"],6)}</small>')
    cells.append(fmt(replay['conditions'][cond]['models']['SELECTED_OR_FALLBACK']['rmse'],6))
    results.append(cells)
success=[]
for cond in COND:
    vals=[cond]
    for summary,method in [(xia,'GEO'),(xia,'LOS'),(xia,'WIDE'),(pair,'Flood'),(xia,'Xia_scale_terminal')]:
        x=summary['metrics'][cond][method]
        vals.append(f'{x["complete"]}/8 · {x["success20"]}/8')
    success.append(vals)
cost_rows=[]
for cond in COND:
    rr=[r for r in records if r['case'].startswith(cond+'_')]
    ks=[r['K'] for r in rr]
    cost_rows.append([cond,fmt(np.mean([r['LOS_seconds'] for r in rr])),fmt(np.mean([r['seconds'] for r in rr])),f'{min(ks)}–{max(ks)}',sum(r['stop']=='no_detected_increment' for r in rr),fmt(replay['conditions'][cond]['equivalent_risk_seconds']/60,2)])

branch_rows=[]
for r in diag['stage_B']['rows']:
    branch_rows.append([r['case'],fmt(r['first_cost'],6),fmt(r['second_cost'],6),r['selected'],fmt(r['first_rmse_m'],6),fmt(r['second_rmse_m'],6)])
recovery=[]
for r in diag['comparisons']:
    s=read(DIAG/(r['case']+'_alternative.json'))
    recovery.append([r['case'],fmt(r['original_rmse_m'],6),fmt(rmse(s['los']['p'],bank['HIGH']['p']),6),fmt(r['alternative_rmse_m'],6),f'{r["original_K"]} → {r["alternative_K"]}',f'{r["original"]["cost"]:.6f} → {r["alternative"]["cost"]:.6f}'])

body=''
body+=section('01 / 先回答疑问','LOS能筛对区域，不等于LOS已经足够精确',
 '<div class="callout"><strong>不能简单归因为“多径不严重”。但当前实验确实包含对定位较有利的假设。</strong><p>LOS模型虽然省略了反射，仍保留跨站传播时延的几何结构。它可以把错误的大区域与真实源附近区分开，同时在真实区域内留下米级或亚米级偏差。筛初值与做最终估计，是两个不同要求。</p></div>'
 '<div class="split"><div><h3>本轮实际验证了什么</h3><p>几何前端已经提出两个位置组合，LOS只负责局部精修和重排。HIGH_R00/R05的原始第26个配对，才是跳过25个同区域近重复配对后得到的第二个区域；不是简单“取前两名”。</p><p>没有验证LOS从全搜索域独立盲搜就能得到同样效果。</p></div><div><h3>场景并非极端多径</h3><p>所有链路有LOS，每一条反射都弱于LOS；路径额外时延较易分辨，站点同步，信道短时不变。普通组总反射功率仅约直达的25.1%。</p><p>HIGH组总反射功率为直达的120%，仍有显著干扰；但它也不是直达缺失、强单条反射或密集不可分辨多径。</p></div></div>'
 '<p><strong>是否“已经够好”，取决于预先确定的精度要求。</strong>若目标只需20 m成功，普通组LOS已足够，增加多径求解未必划算；若关心亚米级RMSE，当前多径处理仍有明确的样本精度收益。不能为突出方法而事后提高精度要求，也不需要先让LOS完全失效才有研究价值。</p>', 'answer')
body+=section('02 / 场景合同','现在究竟在模拟什么',
 '<p>范围：最近11月2—4日的32条条件校准、风险回放、强近邻、重采样与失败归因。更早的G10/G11旧方法与旧场景不混入当前排名。</p>'+
 table(['项目','实际设置','估计器信息／边界'],params)+
 '<div class="split"><div>'+svg+'</div><div>'+table(['对象','x / m','y / m','角色'],coordinate_rows)+'</div></div>'
 '<details><summary>查看每站直达带内SNR与源间功率差</summary>'+table(['银行','源','站0','站1','站2','站3','站4','五站平均功率对应SNR'],[[cond,chr(65+q)]+[fmt(v,3) for v in bank[cond]['los_snr_db'][q]]+[fmt(10*np.log10(np.mean(10**(bank[cond]['los_snr_db'][q]/10))),3)] for cond in ('L0','L3') for q in range(2)])+'<p>单位dB。L0一行同样用于L2/HIGH。这里先平均线性功率再转dB，不是平均五个dB值；没有设定所有链路都6 dB。</p></details>', 'scene')
body+=section('03 / 多径强度','生成设置与算法可获得先验分开列',
 table(['条件','每链路反射 / 总反射','Σ反射功率 / 直达功率','直达/总反射 dB','单条反射/直达功率实际范围','额外时延区间 / μs'],channel_rows)+
 '<p>功率比按路径系数平方之和定义，不等于相干叠加后反射波形的瞬时功率。相位和同源相关性会产生增强或抵消，不能直接从功率和推断所有频率的干扰强弱。</p>'
 '<p>算法<strong>不知道</strong>表中的真实反射数、时延区间和精确功率比。只使用公共信息、同源路径共享波形、非负额外时延及“每条反射不强于直达”的已批准约束。真实生成值只用于本页说明和误差评价。</p>'
 '<div class="note">L3换用了独立波形/信道银行；L0/L2/HIGH沿用同一个银行，但噪声种子不同。因此不能把L2与L3的差值当作“只增加一条反射”的严格因果效应。32条主要检验条件噪声重复性，尚不足以代表跨构型、跨信道泛化。</div>', 'channel')
body+=section('04 / 实施细节','方法不是只换一个名字',table(['方法／角色','如何从观测得到位置','关键参数／约束／停止','本轮证据范围'],methods), 'methods')
body+=section('05 / 数学与代码','共同目标、前端与增径规则',
 '<div class="formula">y<sub>f</sub> = H<sub>f</sub>(p,g,ρ,Δτ)s<sub>f</sub> + n<sub>f</sub><br>H<sub>mq</sub>(f) = g<sub>mq</sub>e<sup>−j2πf‖p<sub>q</sub>−r<sub>m</sub>‖/c</sup> [1 + Σ<sub>ℓ</sub>ρ<sub>mqℓ</sub>e<sup>−j2πfΔτ<sub>mqℓ</sub></sup>]<br>ŝ<sub>f</sub> = H<sub>f</sub><sup>†</sup>y<sub>f</sub>，J = Σ<sub>f</sub>‖y<sub>f</sub>−H<sub>f</sub>ŝ<sub>f</sub>‖² / Σ<sub>f</sub>‖y<sub>f</sub>‖²</div>'
 '<p>两个源在同一频点联合拟合，不是逐频独立定位。所有频点共享位置、路径时延与链路增益。频率相关的未知波形系数被消元。LOS是反射项为空的特例；多径模型维数为20+3K个实参数（K为全系统反射分量数），波形不计入这个显式优化维数。</p>'
 '<p>这是有限记录数据上的逐频传播模型，仍是对有限窗传播的表示近似；没有把旧有限记录精确算子研究重新引入。站0两源直达增益归一为1，吸收公共幅相歧义，不表示知道真实站0增益。</p>'+
 table(['环节','实现数值与规则'],[
 ('互相关','原记录补零到16384点，保留±6.25 MHz；频谱插值16倍；在站间距/c允许范围内取局部峰；抛物线亚采样修正。不是取6个相邻最大采样值。'),
 ('几何候选','枚举非共线三站和TDOA峰组合，代数消元得到二次方程可能根，检查搜索域及方程一致性；不是固定只用三站，也不是Chan–Ho复现。'),
 ('全站验证','10站对，每对找最近时差峰；代价Σmin(距离差残差²,H²)，H=c/B≈23.983 m；最多100次几何更新、20次减半回溯；同一个峰不能给两源重复累计奖励。'),
 ('波形与LOS','FFT norm=ortho；H伪逆rcond=1e−12；TRF最小二乘x_scale=jac，ftol=xtol=1e−12、gtol=1e−10，max_nfev=300；另报告原坐标PG≤1e−7。'),
 ('增径搜索','每步搜索5×2×8191=81,910个源—站—采样延时候选；先投影消去当前波形/参数局部切空间，再评估二维复幅度得分；选最大过阈值项。'),
 ('增径阈值','第t步χ²₂上尾概率为0.05/[t(t+1)·10·(N−1)]；这是当前采用的局部检验/多重搜索控制，不把其理想参考分布声称为整个自适应过程已证明的精确误报保证。'),
 ('多径优化','原reference.fit：缩放SLSQP，ftol=1e−12，原总预算300步，必要时一次重定心；正式入口实际接reference_feasible.fit做可行终态接受及有限数值收尾。'),
 ('约束与接受','位置±3000 m；每条反射实虚部位于单位圆盘；Δτ∈[0,N/Fs]；原坐标PG≤1e−7、可行闭合误差<1e−7；目标不得增大超过1e−10。'),
 ('外层停止','无候选过检验则正常停止；非驻点/非下降则数值停止；达到64总分量为容量停止，不等于未知阶数已充分收敛。没有删除/替换既有路径的算子。'),
 ('风险公式','R̂(k)=‖p̂k−p̂W‖²/2 + σ̂²⟨Lk,LW⟩ − σ̂²‖LW‖²/2；L来自实际非零残差Hessian的局部位置响应；不是只用零残差GN。'),
 ('风险门与选择','仅有限、驻定、内点、曲率/导数检查通过且源标签可对齐的状态可评分；最小风险、平局低阶；参照不适用则回退。负风险不截零；没有用真实位置挑模型。'),
 ])+'<p>这里只描述实际实现，不把通用VP、SLSQP、局部敏感性或多初值重排本身列为原创理论。</p>', 'implementation')
body+=section('06 / 主比较','同一32条观测上的定位RMSE',
 '<p>单位m。每条件8条，两个源先做最小总平方误差匹配，然后对16个源误差平方取平均再开方。<strong>不是逐条RMSE的算术平均。</strong>WIDE为原冻结终态，没有用后来的诊断恢复结果覆盖。</p>'+
 table(['条件','GEO','LOS-DPD','多径DPD WIDE','Flood 固定K适配','Xia 独立随机流程','插件风险选择／回退'],results)+
 '<h3>完整输出与20 m双源成功</h3><p>每格为“完整输出条数 / 8 · 两个源都≤20 m的条数 / 8”。缺失输出不能当成0误差。</p>'+table(['条件','GEO','LOS','WIDE','Flood','Xia'],success)+
 '<p>L2中LOS→WIDE的RMSE降幅约73.1%，L3约58.9%。这是同观测配对的描述性结果，不代表所有多径构型。HIGH原结果被两次公里级误定位主导；不能只展示正常六条而省略失败，也不能用Xia的完整子集RMSE与WIDE全体RMSE直接排名。</p>'
 '<details><summary>历史Xia公共候选适配：与当前独立随机流程不同</summary>'+table(['条件','旧公共候选终态RMSE/m','完整/8','成功/8'],[[c,fmt(pair['metrics'][c]['Xia_terminal']['full_rmse_m'],6) if pair['metrics'][c]['Xia_terminal']['full_rmse_m'] is not None else '仅完整子集：'+fmt(pair['metrics'][c]['Xia_terminal']['conditional_rmse_m'],6),pair['metrics'][c]['Xia_terminal']['complete'],pair['metrics'][c]['Xia_terminal']['success20']] for c in COND])+'<p>历史版本改变了首轮候选分布；后续独立随机版本恢复自己的前端，所以两列不能混用最有利成绩。原生随机1000粒子曾出现覆盖不足，不再作为主要强基线结果。</p></details>', 'results')
body+=section('07 / 计算成本','精度收益付出了多少计算',
 table(['条件','GEO+LOS均值/s','完整多径流程均值/s','最终总K范围','正常停止/8','风险计算等价累计/min'],cost_rows)+
 '<p>GEO+LOS由保存的前端与LOS阶段计时相加，不是重新独立计时；完整流程包含前端、增径、拟合及保存。风险计算等价成本包含复用诊断的原计算时间，不等于该次回放新增加的墙钟。</p>'+
 table(['任务','规模／时间','解释'],[
 ('原完整多径定位32条',f'{replay["overall"]["original_localization_seconds"]/60:.2f} min','四条件共用表中同一批；没有重复累计为多批独立证据'),
 ('插件风险回放',f'{replay["replay_cost"]["seconds"]/3600:.2f} h新增回放账本','没有新定位；包含各中间状态风险诊断与保存'),
 ('Flood＋旧Xia公共候选',f'{pair["total_seconds"]:.2f} s / 32条配对','联合批总时，不当成单个方法平均时间'),
 ('Xia独立随机流程',f'{xia["cost"]["seconds"]:.2f} s / 32条','含数值表准备、评分抽查与保存；异批计时，不作严格速度胜负'),
 ('完整bootstrap48次','43.86 min','拟合世界重定位；本轮没有获得新增选模精度或净加速'),
 ('最新A/B＋C归因',f'{(diag["cost"]["seconds"]+precision["seconds"])/60:.2f} min','两例完整反事实＋八条LOS筛选及数值核验；不是正式新方法32条重跑'),
 ])+'<p>普通条件中，完整多径明显比LOS昂贵。若应用只需米级或20 m定位，是否值得付出该代价必须单独判断；“更准”不能自动等同“更实用”。</p>', 'cost')
body+=section('08 / 最新诊断','LOS筛选与多径精修，各自做了什么',
 table(['案例','原多径终态/m','替代区域LOS/m','替代区域多径终态/m','原K→新K','原J→新J'],recovery)+
 '<p>两次替代分支都来自观测候选，不是真位置起点。原后端不变；新分支正常停止并达到PG门。这里的误差是单条观测内双源RMSE，不是跨样本总体RMSE。两例不能代替新算法的独立验证。</p>'
 '<h3>HIGH全部8条：两个区域的LOS目标排序</h3>'+table(['案例','区域1 J','区域2 J','选中区域','区域1误差/m','区域2误差/m'],branch_rows)+
 '<p>选择只看J，不看误差；误差是事后评价。本批8/8选中误差较小者，所选LOS的合并RMSE约1.746 m。其余六条原完整多径结果仍保留，未把两例恢复拼接成“已独立验证的新方法8/8”。</p>'
 '<details><summary>停止精度核查：通过的是什么，还有什么没通过</summary><p>原16个LOS状态仅5个达到PG≤1e−7；对其余11个各在原剩余300次预算内续算一次后，14/16达到。所选8支全部达到，未选R01/R02第二支仍未达到。排序8/8不变，最大位置变化约8.253 μm。没有放宽阈值，也没有无限续算。</p><p>因此可以说有限预算排序在本批数值上稳定，不能说全部候选均充分优化、更不能说获得全局最优排名。正式“有限预算筛选＋严格最终精修”的部署口径仍待确认。</p></details>'
 '<p>原两条错误轨迹即使用真值事后选最好的状态，也仍约3379 m。风险选模只能选择已有状态，不能把从未进入候选轨迹的正确位置变出来。</p>', 'diagnosis')
body+=section('09 / 风险线回顾','为什么几轮核验后仍没有可靠增益',
 table(['轮次','实际比较','结果','能说明／不能说明'],[
 ('32条端点校准','LOS与完整多径端点，实际Hessian局部风险与真实损失','普通多径端点精度差明确；HIGH适用共同子集很少','粗分“无多径/有多径”不等于精确排好多阶相近模型'),
 ('32条全部轨迹回放','同一保存轨迹内选插件风险最小状态',f'{replay["overall"]["improved"]}改善、{replay["overall"]["harmed"]}变差、{replay["overall"]["unchanged"]}不变；2条回退','平均误差未下降；不是理论最优已知，也不意味着所有风险理论无效'),
 ('固定清单局部响应','14次固定拓扑重拟合＋8次小扰动','小扰动公式通过；完整噪声差分仍有明显失配','问题不能全归于重选路径；没有严格同起点的完整重选支持配对'),
 ('完整有限噪声自举','3个观测拟合donor×16副本；每次完整重定位','48/48正常停止；L2/L3所选cap均落在终态平台','重选路径与非线性重求解没有自动消除参照失配；没有净收益'),
 ('错误区域诊断','两条HIGH换观测区域，保留原后端','两例均恢复，目标更低；原轨迹内无正确位置','候选区域选择与同一区域内风险选模必须分开研究'),
 ])+
 table(['条件','bootstrap推荐cap','旧R01–R07原WIDE RMSE/m','推荐cap RMSE/m','解释'],[[c,boot['conditions'][c]['chosen_cap'],fmt(boot['conditions'][c]['receiver_wide_rmse_m'],6),fmt(boot['conditions'][c]['receiver_chosen_rmse_m'],6),'平局首项；不改变终态输出' if c!='L0' else '小幅变差；未获新增收益'] for c in ('L0','L2','L3')])+
 '<p>本表7次与主表8次分母不同：R00用于拟合生成参照，R01–R07用于迁移评价。48个自举副本不是48个新的物理信道。当前未启动640条，未因负面结果调风险系数。</p>', 'risk')
body+=section('10 / 完整明细','32条原冻结结果，不隐藏长尾',
 '<details><summary>展开32条：几何／LOS／多径误差、阶数、停止与耗时</summary>'+table(['案例','GEO/m','LOS/m','WIDE/m','总K','终止原因','完整耗时/s'],[[r['case'],fmt(r['GEO'],6),fmt(r['LOS'],6),fmt(r['WIDE'],6),r['K'],'容量触顶' if r['stop']=='capacity' else '无显著新增分量',fmt(r['seconds'],3)] for r in records])+'</details>', 'records')
source_links=[
 (BASE/'risk_calibration_20261102/执行协议.md','32条数据协议'),
 (CAL/'contract.json','冻结数据银行、版本与输入合同'),
 (BASE/'unknown_order_20261030/cross_order.py','生成与完整流程实现'),
 (BASE/'finite_iq_clues.py','BPSK/RRC与互相关实现'),
 (BASE/'bounded_consensus.py','五站共识与联合候选评分'),
 (BASE/'unknown_order_20261030/reference.py','多径响应、增径检验、优化'),
 (BASE/'unknown_order_20261030/reference_feasible.py','正式数值收尾实现'),
 (XIA/'summary.json','Xia独立流程及原冻结成绩'),
 (BASE/'xia_scale_20261103/尺度适配结果与下一步.md','Xia适配范围与失败分析'),
 (PAIR/'summary.json','Flood与旧Xia适配结果'),
 (REPLAY/'summary.json','32条风险完整回放结果'),
 (BASE/'risk_support_control_20261103/run_20261103_165036_887173/固定清单局部响应报告.md','固定清单核验与未实施部分'),
 (BOOT/'final_analysis.json','48次自举最终分析'),
 (DIAG/'result.json','两例完整归因与八条LOS比较'),
 (PREC/'result.json','共同精度单列诊断'),
 (BASE/'failure_attribution_20261104/结果与下一决策.md','最新研究判断与下一审批点'),
]
body+=section('11 / 追溯与边界','所有表格回到保存证据',
 '<div class="sources">'+''.join(link(p,label) for p,label in source_links)+'</div>'
 '<p>本页仅整理与重算保存证据，不运行定位，不更改输入、算法、阈值或正式输出。两例诊断恢复与八条筛选不覆盖原冻结成绩。论文贡献、预测机制与优化表现分别评价；不从当前数据推断全局可辨识、普适鲁棒或期刊录用。</p>'
 '<p>生成脚本与核验：<a href="build_report.py">build_report.py</a> · <a href="evidence.json">结构化数值与来源SHA</a> · <a href="核验说明.md">统计与口径检查</a>。无需网络、无需外部字体或脚本。</p>', 'sources')

css='''
:root{--ink:#193440;--muted:#5d7078;--teal:#126b6e;--paper:#f5f5f0;--line:#dbe3df;--warm:#ad6337}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:70px}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}header,main,footer{max-width:1250px;margin:auto;padding:28px}header{padding-top:54px;padding-bottom:32px}h1{font-size:clamp(29px,4vw,45px);line-height:1.3;margin:14px 0}h2{font-size:27px;line-height:1.45;margin:6px 0 20px}h3{font-size:19px}p{margin:14px 0}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:var(--teal)}.lede{max-width:880px;color:var(--muted);font-size:18px}.pills{display:flex;flex-wrap:wrap;gap:9px}.pills span{background:#e4eeeb;color:#285b5d;padding:4px 13px;border-radius:20px;font-size:13px}nav{position:sticky;top:0;background:#f5f5f0f2;backdrop-filter:blur(12px);border-block:1px solid var(--line);z-index:3}nav div{max-width:1250px;margin:auto;padding:10px 28px;display:flex;gap:21px;overflow:auto;white-space:nowrap}a{color:var(--teal);text-underline-offset:3px}nav a{text-decoration:none;font-size:14px}section{background:white;padding:32px;border:1px solid var(--line);border-radius:18px;margin-bottom:24px;box-shadow:0 3px 18px #18344004}.callout{background:#eaf3ef;border-left:4px solid var(--teal);padding:20px 24px;border-radius:5px}.note{background:#faf2e8;border-left:4px solid #c9854d;padding:15px 20px;border-radius:4px}.split{display:grid;grid-template-columns:1fr 1fr;gap:26px;align-items:center}.table-wrap{width:100%;overflow:auto;margin:18px 0;border:1px solid var(--line);border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7}th{background:#edf3f0;text-align:left;color:#28545a;font-size:13px}td,th{padding:12px 14px;vertical-align:top;border-bottom:1px solid #e4eae7;min-width:90px}td:first-child{font-weight:600}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafbf9}tbody tr:hover{background:#f0f6f4}small{color:var(--muted);font-size:12px}details{margin-top:18px;border:1px solid var(--line);border-radius:10px;padding:16px 20px}summary{cursor:pointer;color:var(--teal);font-weight:600}svg{width:100%;font:14px "Microsoft YaHei",sans-serif;fill:#365d68}.formula{overflow:auto;background:#f5f7f6;border-radius:8px;padding:22px;font:16px/2.3 "Cambria Math","Segoe UI",serif}.sources{display:grid;grid-template-columns:1fr 1fr;gap:10px}.sources a{padding:8px 12px;background:#f3f7f5;border-radius:6px;font-size:14px}footer{color:var(--muted);font-size:13px}button{background:white;color:var(--teal);border:1px solid var(--line);padding:7px 14px;border-radius:7px;cursor:pointer}@media(max-width:700px){header,main,footer{padding:18px}header{padding-top:30px}section{padding:20px 16px}.split,.sources{grid-template-columns:1fr}h2{font-size:24px}td,th{min-width:110px;padding:10px}nav div{padding:10px 18px}.formula{font-size:14px}}@media print{nav,button{display:none}body{background:white}header,main{max-width:none;padding:0}section{break-inside:auto;border:0;padding:14px 0;box-shadow:none}table{font-size:10px}.table-wrap{overflow:visible}a{color:inherit}details{border:0}.split{display:block}}
'''
css += '.split>*{min-width:0}.sources>*{min-width:0;overflow-wrap:anywhere}section{min-width:0}p{overflow-wrap:anywhere}'
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>当前实验场景、方法与结果｜2026-11-04</title><style>'+css+'</style></head><body><header><div class="eyebrow">MULTIPATH DPD · EVIDENCE ATLAS · 2026.11.04</div><h1>当前实验场景、方法与结果</h1><p class="lede">把“在哪里找”“怎样精修”“选择哪个模型”分开看：一页查清实际参数、完整流程、最近几轮成绩，以及LOS筛选为何有用。</p><div class="pills"><span>5站 · 双源 · 2D RF</span><span>未知波形 / 未知路径数</span><span>32条条件观测</span><span>只回读，不新增实验</span></div></header><nav><div>'+''.join(f'<a href="#{i}">{t}</a>' for i,t in [('answer','先看结论'),('scene','场景参数'),('channel','多径强度'),('methods','方法实现'),('results','结果比较'),('cost','计算成本'),('diagnosis','LOS诊断'),('risk','风险线'),('sources','证据')])+'</div></nav><main>'+body+'</main><footer>截至2026-11-04的保存证据。建议先看第1、6、8节，再按需展开细节。<button onclick="document.querySelectorAll(\'details\').forEach(x=>x.open=true)">展开所有细节</button> <button onclick="window.print()">打印 / 保存PDF</button></footer></body></html>'
assert all(CHECKS.values()), CHECKS
for p,h in SOURCES.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
evidence = dict(status='ANALYZED',new_localization_runs=0,checks=CHECKS,source_hashes=SOURCES,
                metrics_recomputed=metrics,per_record=records,latest_diagnostic=diag['stage_B'],
                scene_conditions=channel_rows,formal_results_replaced=False)
(HERE/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
(HERE/'实验场景方法与结果.html').write_text(page,encoding='utf-8')
print(json.dumps(dict(report=str(HERE/'实验场景方法与结果.html'),checks=len(CHECKS),source_files=len(SOURCES),tables=page.count('<table>')),ensure_ascii=False))
