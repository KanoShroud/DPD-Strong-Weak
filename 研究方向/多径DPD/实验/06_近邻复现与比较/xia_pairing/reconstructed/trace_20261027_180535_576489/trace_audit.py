"""Offline stage audit; no new location solve or parameter change.

Truth probes are diagnostic only and never supplied to a decoder. Fixed radii
20/100/400 m describe coverage, not new success criteria. Local probes use a
25 m lattice inside a 100 m disk about each truth, including its exact center.
"""
import csv
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from pilot import N, ROOT, observe, sha, write, xia

RUN = ROOT / 'pilot_20261027_170421_805496'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    return dict(n=len(v), rmse_m=float(np.sqrt(np.mean(v*v))) if len(v) else None,
                median_m=float(np.median(v)) if len(v) else None,
                within20=int(np.sum(v <= 20)), within100=int(np.sum(v <= 100)))


def clean_decoder(out):
    return json.loads(json.dumps(out, default=lambda a: a.tolist()))


def main():
    start = time.perf_counter()
    out = ROOT / ('trace_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    source = RUN.resolve(strict=True)
    assert source.is_relative_to(ROOT.resolve())
    manifest = read(source/'preflight.json')
    for p, digest in manifest['sources'].items():
        assert sha(source/Path(p).name) == digest
        assert sha(Path(p)) == digest, 'Live scoring dependency changed'
    evidence = list(source.glob('*.json')) + list(source.glob('input_*.npz'))
    hashes = {p.name: sha(p) for p in evidence}
    out.mkdir(exist_ok=False)
    write(out/'protocol.json', {
        'input': str(source), 'input_hashes': hashes, 'script_sha': sha(__file__),
        'no_new_solver': True, 'no_changed_algorithm': True,
        'coverage_radii_m': [20,100,400], 'local_probe_radius_m': 100,
        'local_probe_step_m': 25, 'truth_use': 'offline diagnosis only',
        'first_target_identity': 'source A fixed; no permutation or nearest-source remapping',
        'illustrations': 'first sorted case that fails immediately, and first that loses a complete pair',
        'source_paper': 'Xia 2021 pp2856-2857 Eq9-16,20; source A RMSE'})
    (out/'trace_audit.py').write_bytes(Path(__file__).read_bytes())
    records = [read(p) for p in sorted(source.glob('r*_G*.json'))]
    assert len(records) == 84
    cases, rounds, neighborhoods, probes = [], [], [], []
    cache = {}
    checks = 0
    offsets = np.array([(x,y) for x in range(-100,101,25)
                        for y in range(-100,101,25) if x*x+y*y <= 10000], float)
    assert len(offsets) == 49
    exemplars = {}
    for rec in records:
        i = rec['realization']
        if i not in cache:
            cache[i] = dict(np.load(source/f'input_{i:02d}.npz'))
        z = cache[i]
        assert sha(source/f'input_{i:02d}.npz') == rec['input_sha']
        y, _, _ = observe(z, rec['snr_db'], rec['sir_db'], rec['paths'])
        assert hashlib.sha256(y.tobytes()).hexdigest() == rec['observation_sha']
        checks += 2
        name = f'r{i:02d}_snr{rec["snr_db"]:g}_sir{rec["sir_db"]:g}_G{rec["paths"]}'
        base = dict(case=name, realization=i, G=rec['paths'], SNR=rec['snr_db'], SIR=rec['sir_db'])
        qp = z['truth'][:,None,:] + offsets[None]
        local = xia.scores(qp.reshape(-1,2), z['stations'], y, z['frequency'], N).reshape(2,-1)
        center = np.flatnonzero(np.all(offsets == 0, axis=1))[0]
        probes.append(dict(case=name, points=qp, scores=local))
        per_case = []
        for step in rec['output']['steps']:
            pts, val = np.asarray(step['points']), np.asarray(step['scores'])
            # Replay every saved score and the original decoder, without new particles.
            again = xia.scores(pts, z['stations'], y, z['frequency'], N)
            assert np.allclose(again, val, rtol=1e-12, atol=0)
            replay = xia.decode(pts, val, 2, rec['output']['delta_h_m'])
            assert clean_decoder(replay) == step['output']
            checks += 2
            filt = xia.threshold(val)
            keep = filt['keep']
            idx = int(val.argmax())
            first = pts[idx] if keep.any() else None
            first_a = float(np.linalg.norm(first-z['truth'][0])) if first is not None else None
            first_b = float(np.linalg.norm(first-z['truth'][1])) if first is not None else None
            trace = replay.get('trace', [])
            radius = trace[-1][0] if trace else None
            dfirst = np.linalg.norm(pts-first,axis=1) if first is not None else None
            rr = dict(**base, round=step['round'], status=replay['status'],
                      retained=int(keep.sum()), modal_bin=filt.get('modal_bin'),
                      threshold=filt.get('threshold'), maximum=float(val.max()),
                      first_A_error_m=first_a, first_B_error_m=first_b,
                      first_nearer_B=bool(first_b < first_a) if first is not None else None,
                      raw_argmax_A_error_m=float(np.linalg.norm(pts[idx]-z['truth'][0])),
                      cluster_radius_m=radius,
                      pair_complete=replay.get('positions') is not None)
            rounds.append(rr)
            per_case.append(rr)
            for q in range(2):
                d = np.linalg.norm(pts-z['truth'][q],axis=1)
                for rad in [20,100,400]:
                    near = d <= rad
                    survived = near & keep
                    outside = survived & (dfirst > radius) if radius is not None else np.zeros(len(pts),bool)
                    nr = dict(**base, round=step['round'], target='A' if q==0 else 'B', radius_m=rad,
                              nearest_m=float(d.min()), particles=int(near.sum()),
                              survived=int(survived.sum()), outside_first_cluster=int(outside.sum()),
                              truth_score=float(local[q,center]),
                              truth_rank_vs_particles=int(np.sum(val>local[q,center]))+1,
                              local100_best_score=float(local[q].max()),
                              local100_best_rank=int(np.sum(val>local[q].max()))+1,
                              truth_above_threshold=bool(local[q,center]>=filt.get('threshold',np.inf)),
                              local100_above_threshold=bool(local[q].max()>=filt.get('threshold',np.inf)))
                    if not near.any(): label = 'NO_PARTICLE_IN_NEIGHBORHOOD'
                    elif not survived.any(): label = 'ALL_NEARBY_PARTICLES_FILTERED'
                    elif radius is None: label = 'SURVIVORS_WITHOUT_CLUSTER'
                    elif not outside.any(): label = 'NEARBY_SURVIVORS_INSIDE_FIRST_CLUSTER'
                    else: label = 'NEARBY_SURVIVORS_OUTSIDE_FIRST_CLUSTER'
                    nr['descriptive_stage'] = label
                    neighborhoods.append(nr)
        valid_first = [r for r in per_case if r['first_A_error_m'] is not None]
        c = dict(**base, rounds=len(per_case), stop=per_case[-1]['status'],
                 first_round_A_error_m=per_case[0]['first_A_error_m'],
                 terminal_first_A_error_m=per_case[-1]['first_A_error_m'],
                 last_available_first_A_error_m=valid_first[-1]['first_A_error_m'] if valid_first else None,
                 last_available_first_round=valid_first[-1]['round'] if valid_first else None,
                 ever_complete=any(r['pair_complete'] for r in per_case),
                 first_round_raw_max_A_error_m=per_case[0]['raw_argmax_A_error_m'])
        cases.append(c)
        key = 'lost_pair' if c['ever_complete'] else 'first_round_failure'
        if key not in exemplars: exemplars[key] = (rec,z,base)
        print(f'离线链路核对 {len(cases)}/84 | {name}', flush=True)
    assert all(sha(source/name)==h for name,h in hashes.items())
    checks += len(hashes)
    csv_write(out/'rounds.csv', rounds)
    csv_write(out/'neighborhoods.csv', neighborhoods)
    csv_write(out/'cases.csv', cases)
    write(out/'truth_probes.json', probes)
    grouped = []
    for g,snr,sir in sorted(set((r['G'],r['SNR'],r['SIR']) for r in cases)):
        sub = [r for r in cases if (r['G'],r['SNR'],r['SIR'])==(g,snr,sir)]
        grouped.append(dict(G=g,SNR=snr,SIR=sir,total=len(sub),
            initial_A=stats([r['first_round_A_error_m'] for r in sub]),
            terminal_A=stats([r['terminal_first_A_error_m'] for r in sub]),
            last_available_A=stats([r['last_available_first_A_error_m'] for r in sub])))
    summary = dict(calls=84, independent_realizations=12, scored_rounds=len(rounds),
        first_initial=stats([r['first_round_A_error_m'] for r in cases]),
        first_terminal=stats([r['terminal_first_A_error_m'] for r in cases]),
        first_last_available=stats([r['last_available_first_A_error_m'] for r in cases]),
        grouped=grouped,
        failures_by_round={str(j):dict(Counter(r['status'] for r in rounds if r['round']==j)) for j in [1,2,3]},
        nearest_initial_target={t:stats([r['nearest_m'] for r in neighborhoods if r['round']==1 and r['target']==t and r['radius_m']==100]) for t in ['A','B']},
        initial100_stage={t:dict(Counter(r['descriptive_stage'] for r in neighborhoods if r['round']==1 and r['target']==t and r['radius_m']==100)) for t in ['A','B']},
        initial_probe={t:{'truth_above_threshold':sum(r['truth_above_threshold'] for r in neighborhoods if r['round']==1 and r['target']==t and r['radius_m']==100),
                          'local100_above_threshold':sum(r['local100_above_threshold'] for r in neighborhoods if r['round']==1 and r['target']==t and r['radius_m']==100),
                          'local100_top10':sum(r['local100_best_rank']<=10 for r in neighborhoods if r['round']==1 and r['target']==t and r['radius_m']==100)} for t in ['A','B']},
        first_nearer_B_initial=sum(r['first_nearer_B'] is True for r in rounds if r['round']==1),
        ever_complete=sum(r['ever_complete'] for r in cases),
        checks=checks, seconds=time.perf_counter()-start,
        no_new_solves=True, original_inputs_unchanged=True)
    write(out/'summary.json', summary)
    write(out/'audit.json', dict(checks=checks, passed=True, original_inputs_unchanged=True,
                                source_paper_sha=manifest['paper_sha'], script_sha=sha(__file__)))
    for key,(rec,z,base) in exemplars.items():
        write(out/(key+'.json'), dict(case=base, stations=z['stations'], truth=z['truth'],
                                     output=rec['output']))
    render(out, summary, rounds, neighborhoods, exemplars)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('REPORT',str(out),flush=True)


def render(out,s,rounds,neighborhoods,exemplars):
    def table(headers, data):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table></div>'
    byround = table(['轮次','进入本轮','完整双源','筛空','第二簇缺失'],
        [[j,sum(r['round']==j for r in rounds),s['failures_by_round'][str(j)].get('OK',0),
          s['failures_by_round'][str(j)].get('NO_RETAINED_PARTICLE',0),
          s['failures_by_round'][str(j)].get('EMPTY_SECOND_CLUSTER',0)] for j in [1,2,3]])
    firsttab = table(['口径','可计算/84','固定目标A的RMSE/m','中位数/m','20m内'],
        [[label,f'{s[k]["n"]}/84',f'{s[k]["rmse_m"]:.2f}',f'{s[k]["median_m"]:.2f}',s[k]['within20']]
         for label,k in [('第一轮第一峰','first_initial'),('实际终止轮第一峰','first_terminal'),('最近可用第一峰（离线保护）','first_last_available')]])
    groups = table(['G','SNR','发射SIR','第一轮A RMSE / 分母','终轮A RMSE / 分母'],
        [[r['G'],r['SNR'],r['SIR'],f'{r["initial_A"]["rmse_m"]:.1f} / {r["initial_A"]["n"]}',
          f'{r["terminal_A"]["rmse_m"]:.1f} / {r["terminal_A"]["n"]}'] for r in s['grouped']])
    nt = table(['源','初始100m内有点','有点但被全删','真实点通过阈值','100m局部探针至少一个通过','局部探针最佳排名≤10'],
        [[t,84-s['initial100_stage'][t].get('NO_PARTICLE_IN_NEIGHBORHOOD',0),s['initial100_stage'][t].get('ALL_NEARBY_PARTICLES_FILTERED',0),s['initial_probe'][t]['truth_above_threshold'],s['initial_probe'][t]['local100_above_threshold'],s['initial_probe'][t]['local100_top10']] for t in ['A','B']])
    examples = ''
    for key,(rec,z,base) in exemplars.items():
        examples += '<h3>'+('曾有两个位置，后来丢失' if key=='lost_pair' else '第一轮就无法形成两个位置')+'：'+base['case']+'</h3>'
        examples += '<div class="maps">'
        for step in rec['output']['steps']:
            p=np.asarray(step['points']); v=np.asarray(step['scores']); mask=xia.threshold(v)['keep']
            def xy(q): return (30+(q[0]+3000)/6000*330,360-(q[1]+3000)/6000*330)
            svg='<svg viewBox="0 0 400 405" role="img" aria-label="保存候选点、真实源及筛选结果"><rect x="30" y="30" width="330" height="330" fill="#fafafa" stroke="#b9c7ce"/>'
            for idx,q in enumerate(p):
                x,y=xy(q); svg+=f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{2 if mask[idx] else 1}" fill="{"#d07827" if mask[idx] else "#c5d1d6"}"/>'
            for idx,q in enumerate(z['stations']):
                x,y=xy(q); svg+=f'<rect x="{x-3:.2f}" y="{y-3:.2f}" width="6" height="6" fill="#364f67"/>'
            for idx,q in enumerate(z['truth']):
                x,y=xy(q); svg+=f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="none" stroke="#087f71" stroke-width="2"/><text x="{x+9:.2f}" y="{y:.2f}" font-size="13" fill="#087f71">{["A","B"][idx]}</text>'
            x,y=xy(p[v.argmax()]); svg+=f'<path d="M{x-5},{y-5} l10,10 M{x-5},{y+5} l10,-10" stroke="#a53138" stroke-width="2"/>'
            svg+=f'<text x="30" y="20" font-size="14">第{step["round"]}轮 · 保留{mask.sum()}点</text><text x="30" y="385" font-size="12">x、y范围：−3000～3000 m；红叉：原始最高分</text></svg>'
            examples+='<div>'+svg+'</div>'
        examples+='</div>'
        chosen = [r for r in rounds if r['case']==base['case']]
        examples += table(['轮次','最高分','阈值','众数箱（1—10）','保留点','第一簇半径/m','状态'],
                          [[r['round'],f'{r["maximum"]:.3g}',f'{r["threshold"]:.3g}',r['modal_bin'],r['retained'],r['cluster_radius_m'],r['status']] for r in chosen])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Xia评分到输出：逐轮核查</title><style>
body{max-width:1120px;margin:auto;padding:30px 22px;background:#f4f4ef;color:#203b47;font:16px/1.85 "Microsoft YaHei",system-ui}section{background:white;border:1px solid #dce3e5;border-radius:14px;padding:24px;margin:20px 0}h1{font-size:32px;line-height:1.45}h2{font-size:23px}h3{font-size:17px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #dfe7e9}th{background:#edf3f3}.note{background:#fff2dc;border-left:4px solid #c58b31;padding:14px}.maps{display:flex;gap:12px;flex-wrap:wrap}.maps>div{flex:1;min-width:260px}svg{width:100%;height:auto}.muted{color:#657b84}a{color:#007e73}summary{cursor:pointer;font-weight:bold}</style>
<p class="muted">2026-10-27 · 保存结果逐轮核查 · 没有新定位求解</p><h1>不是只丢了第二个输出：<br>第一峰的定位也没有跑通</h1>
<section><h2>这一轮做了什么</h2><p>回读原84次求解的所有候选、评分、阈值及分组，重新计算原评分和解码，确认与保存状态一致。没有改参数、没有新增粒子或精修；另在真实位置及其100m邻域计算原评分，仅用于事后诊断，不反馈给算法。</p><p>84次来自12个独立基础实现，跨噪声/功率/路径条件配对，不是84个独立构型。本页指标只描述这个重建版本。</p></section>
<section><h2>1．两个位置到底在哪一轮丢了？</h2>'''+byround+'''
<p>“前面、后面”就是同一次求解的第1/2/3轮，不是不同实验批次。每轮都重新计算阈值；两个旧中心并不会被保证保留。</p><p>若众数箱编号为b，原阈值可写成 T=min+(b/5)(max−min)。所以b=5时只保留最高分端点，b≥6时阈值真正超过最高分。这是原规则作用于当前评分分布的结果，不是浮点端点错误。</p></section>
<section><h2>2．只评价论文关心的第一个目标，是否其实很好？</h2>'''+firsttab+'''
<p>第一峰固定与真实源A比较，不使用双源匹配、不改配到更近的真实源。终轮缺第一峰的样本不伪造误差；“最近可用”是离线输出保护，不能写成原论文终态性能。</p><p class="note">之前的25例条件RMSE是在完整双源匹配后计算的；本表是固定目标A的第一峰误差。两者评价对象不同，不能直接比较数值变化。如今补齐第一峰后，表现仍远非论文中所报告的精度。</p><details><summary>按原条件分别查看</summary>'''+groups+'''</details></section>
<section><h2>3．覆盖、评分、筛选：不是只有一个环节有问题</h2>'''+nt+'''
<p>上表均以84次第一轮为分母。100m用于查看候选覆盖，不是修改20m成功门槛。另保存了20/400m统计，避免只用一个半径解释。局部探针为以真值为中心、半径100m、间距25m的49个固定位置；“最佳排名”只相对本轮1000个粒子，不是连续空间全局峰排名。</p><p>真实附近没有粒子时，存在覆盖限制；附近有粒子却被全部删除时，有筛选损失。但即使给真实邻域额外的事后探针，许多样本的评分也不突出。因此不能预先断定，只换聚类就能恢复论文效果。</p><p>“保留点落在第一簇里”也不自动意味着两个真实源被合并：第一簇中心本身可能已经很远。几何计数只是现象分类，不是已证明的物理根因。</p></section>
<section><h2>4．把“前面有、后面没”画出来</h2><p class="muted">灰点：被删除候选；橙点：保留候选；蓝方块：接收站；绿圈A/B：真实源；红叉：原始最高分。案例按文件名顺序选择第一例相应事件，不按定位成绩挑图。</p>'''+examples+'''</section>
<section><h2>5．下一步只查一件事</h2><p><b>优先查明原始评分为什么没有稳定指向目标，而不是直接改阈值或增加重复次数。</b></p><p>原式(9)包含对参考站频谱的除法。当某些频点的参考信号很小，这些频点可能产生很大的比值；两个源的贡献也可能在参考站相互抵消。这是需要检验的线索，本轮没有把它认定为根因。</p><p>建议下一步固定现有输入，拆开式(9)各频点对真值和错误峰的贡献，核对是否由低参考功率频点主导，再检查原文隐含的统计平均/频率定义。若必须改评分或引入正则化，单列适配版本，不悄悄改原复现。当前不启动1000次重复或新方法性能比较。</p></section>
<section><h2>审计与原始表</h2><p>'''+f'{s["checks"]}项复核通过；{s["scored_rounds"]}个已保存轮次；耗时{s["seconds"]:.1f}s。'+'''</p><p><a href="protocol.json">范围及输入身份</a> · <a href="audit.json">审计</a> · <a href="summary.json">精确汇总</a> · <a href="cases.csv">逐例第一目标</a> · <a href="rounds.csv">逐轮筛选</a> · <a href="neighborhoods.csv">邻域归因</a> · <a href="truth_probes.json">真值探针</a> · <a href="trace_audit.py">完整核验脚本</a> · <a href="../公开信息重建报告.html">上一轮报告</a></p><p>原文依据：Xia等，IEEE Systems Journal 2021，pp.2856–2857，式(9)—(16)、式(20)。原文的第一目标统计不等于完整双源输出率。</p></section></html>'''
    (out/'逐轮链路核查.html').write_text(page,encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
