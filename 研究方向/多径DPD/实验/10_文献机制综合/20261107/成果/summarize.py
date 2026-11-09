"""Build source coverage, descriptive paired report and a self-contained HTML."""
import csv
import hashlib
import html
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'AGENTS.md').exists())
BASE = ROOT/'研究方向/多径DPD'
RAW = HERE.parent/'运行数据/branches18'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def rms(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def inline(s):
    s = re.sub(r'<!--.*?-->', '', s)
    s = html.escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', s)


def render(text):
    pieces = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        if lines[0].startswith('#'):
            mark, title = lines[0].split(' ', 1)
            level = min(len(mark)+1, 5)
            pieces.append(f'<h{level}>{inline(title)}</h{level}>')
        elif all(line.startswith('    ') for line in lines):
            pieces.append('<pre>'+html.escape('\n'.join(x[4:] for x in lines))+'</pre>')
        elif lines[0].startswith('|'):
            cells = [x.strip().strip('|').split('|') for x in lines]
            head = '<tr>'+''.join('<th>'+inline(x.strip())+'</th>' for x in cells[0])+'</tr>'
            body = ''.join('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in row)+'</tr>' for row in cells[2:])
            pieces.append('<div class="scroll"><table>'+head+body+'</table></div>')
        elif lines[0].startswith('- '):
            pieces.append('<ul>'+''.join('<li>'+inline(x[2:])+'</li>' for x in lines)+'</ul>')
        else:
            pieces.append('<p>'+inline(block).replace('\n', '<br>')+'</p>')
    return '\n'.join(pieces)


def main():
    data = read(RAW/'rows.json')
    done = read(RAW/'completion.json')
    assert done['count'] == len(data) == 18 and done['source_unchanged']
    pre = read(RAW/'preflight.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in pre['source_hashes'].items())
    assert all(hashlib.sha256((RAW/'inputs'/r['tag']/'iq.npz').read_bytes()).hexdigest() == r['input_sha'] for r in data)
    rows = []
    for d in data:
        a, b = d['branches']['first'], d['branches']['second']
        rows.append(dict(case=d['tag'], condition=d['condition'],
            original_rmse=rms(d['original']['errors_m']), first_rmse=rms(a['errors_m']),
            second_rmse=rms(b['errors_m']), first_cost=a['cost'], second_cost=b['cost'],
            first_K=a['K'], second_K=b['K'], first_normal=a['normal'], second_normal=b['normal'],
            choice=d['pareto_choice'], seconds=d['seconds_including_repeated_front']))
    with (HERE/'两分支可行性.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for condition in ['L0', 'L1_P075', 'L2_P075']:
        rr = [r for r in rows if r['condition'] == condition]
        groups.append(dict(condition=condition, n=len(rr),
            original_rmse=rms([r['original_rmse'] for r in rr]),
            oracle_best_two_rmse=rms([min(r['first_rmse'], r['second_rmse']) for r in rr]),
            first_normal=sum(r['first_normal'] for r in rr),
            second_normal=sum(r['second_normal'] for r in rr),
            decisions=sum(r['choice'] != 'unresolved' for r in rr),
            mean_seconds=float(np.mean([r['seconds'] for r in rr]))))
    summary = dict(groups=groups, cases=rows, algebra=read(RAW/'algebra.json'),
        completion=done, input_and_source_identity_rechecked=True,
        scope='old_development_cases_not_independent_validation',
        oracle_best_two_is_not_algorithm_or_theoretical_bound=True,
        final_gate='HOLD_NO_LARGER_BRANCH_SEARCH_OR_FORMAL_BATCH')
    write(HERE/'summary.json', summary)
    structure = read(HERE.parent/'运行数据/structure_check/result.json')
    write(HERE/'结构边界.json', {k:v for k,v in structure.items() if k != 'source_hashes'})
    corpus = read(BASE/'文献/近邻重审_20261105/文献矩阵.json')
    route_ids = {
        'A 分支与几何': [2,8,9,10,11,12,18,21,22,31,47,53,55,56],
        'B 关联交换': [1,8,11,12,26,31,39,46,49,51,54,58],
        'C 几何共享波形分解': [3,4,5,6,20,21,22,23,26,28,35,41,42,43,44,49,50,54,55,56,59,61,62,63,64,65,69],
        'D 共同子空间': [13,14,15,16,17,19,24,27,28,29,30,33,34,40,45,48,60,61,64,68,69],
        'E 预测与统计选择': [4,18,32,50,57,59,65,66,67],
        'F 求解组织': [1,3,4,5,7,36,37,38,44,46,49,51,57,58,66,67],
        '场景与信息边界': [25,52],
    }
    coverage = []
    for p in corpus:
        k = int(p['id'][1:])
        routes = [name for name, ids in route_ids.items() if k in ids]
        assert routes, p['id']
        r = p['close_reading']
        coverage.append(dict(id=p['id'], title=p['title'], url=p.get('url') or 'https://doi.org/'+p['doi'],
            routes=routes, priors=r['observations_priors'], method=r['method'],
            theory=r['theory'], insight=r['insight'], evidence=r['evidence'],
            scope='existing_close_reading_resynthesized_not_new_fulltext_read'))
    assert len(coverage) == len({p['id'] for p in coverage}) == 69
    write(HERE/'文献机制覆盖.json', coverage)
    lines = ['# 两分支最小核验结果', '',
        '18条旧开发观测，六构型×三种条件，每构型每条件只有一个实现。两支分别盲运行原MP流程，不含真值候选选择。', '',
        '| 条件 | 原MP RMSE(m) | 两支事后择优RMSE(m)，仅诊断 | 第一/第二支正常数 | 无权重支配可决数 | 完整平均秒数 |',
        '|---|---:|---:|---|---:|---:|']
    for g in groups:
        lines.append(f"| {g['condition']} | {g['original_rmse']:.6f} | {g['oracle_best_two_rmse']:.6f} | {g['first_normal']}/6，{g['second_normal']}/6 | {g['decisions']}/6 | {g['mean_seconds']:.3f} |")
    lines += ['', '**没有形成一个覆盖18条的自动选择器。** 未决案例不删除，也不只报告已决子集RMSE。事后择优只表示这两个输出中较好者的经验成绩，不是理论最优界；连它都未恢复的案例，不能靠这两支之间换评分解决。', '',
        '两支合计成本包含重复前端；本轮没有实现缓存加速。原MP数字来自上一轮同观测结果，不把不同批次墙钟差解释成严格提速比。非正常但有限的位置输出全部保留；程序正常退出也不等于求解器全部收敛。', '',
        '结果不支持把“延迟LOS裁决、让已有两支各自增径”直接定为最终方法。按合同停止扩大分支数和调权重；它也没有否定全部分支探索、源—路径交换或结构化分解路线。', '',
        '公式单元检查：', '',
        f"- VP与3阶子式恒等式最大相对误差：{summary['algebra']['minors_vp_max_relative_error']:.3g}。",
        f"- 可逆混合下子空间残差最大相对差：{summary['algebra']['mixing_invariance_max_relative_error']:.3g}。",
        '- 双源交叉关系构造的非零交叉项为4，代数回代误差为0。',
        '- 嵌套子空间例子中，LOS的A/B残差为1/0.5，扩展后为0/0.5；仅用于否定一般安全裁剪推理，不声称它就是当前物理信道。', '',
        '检查：36次目标重算、PG记录、输入与源码身份复核完成；完整原始决策在运行数据/branches18。未修改公共算法，未动独立重复和正式大批。']
    result_text = '\n'.join(lines)+'\n'
    (HERE/'核验结果.md').write_text(result_text, encoding='utf-8')
    cards = []
    for p in coverage:
        cards.append('<details><summary>'+html.escape(p['id']+' · '+p['title'])+'</summary><p class="tag">'+html.escape(' / '.join(p['routes']))+'</p>'+
            ''.join('<p><strong>'+label+'：</strong>'+html.escape(p[key])+'</p>' for key,label in [('priors','原条件'),('method','原方法'),('theory','理论边界'),('insight','已有精读启示'),('evidence','原文定位')])+
            '<a href="'+html.escape(p['url'], quote=True)+'">正式/已登记文献入口</a></details>')
    synthesis = (HERE/'机制综合.md').read_text(encoding='utf-8')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>文献思想与候选方法</title><style>
body{margin:0;background:#f4f5f1;color:#20353d;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:36px 28px}header{background:#153e47;color:white;padding:32px;border-radius:18px}h1{font-size:32px;line-height:1.4;margin:8px 0}h2{margin-top:36px;border-bottom:1px solid #cad6d2;padding-bottom:10px}h3{color:#09656b}nav{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0}a{color:#086b78}header a{color:#bdece4}.card,details{background:white;border:1px solid #d9e1dc;border-radius:12px;padding:20px;margin:15px 0}.notice{background:#fff2d9;padding:18px;border-left:4px solid #bd8427}summary{cursor:pointer;font-weight:600}.tag{font-size:14px;color:#5e727a}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #dce4df;padding:10px;text-align:left;vertical-align:top}th{background:#e5efeb}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0ef;padding:18px;border-radius:8px;font-size:14px}code{background:#eaf0ef;padding:2px 4px}li{margin:7px 0}footer{font-size:13px;color:#65777b;margin-top:30px}@media(max-width:600px){main{padding:16px}header{padding:20px}h1{font-size:25px}}</style><main><header><div>多径DPD · 第10轮 · 2026-11-07</div><h1>从文献思想到候选方法</h1><p>69篇记录综合 → 六条机制路线 → 公式检查 → 18条盲双分支核验</p><p>现有MP保留；尚未获得可冻结的新架构，不把负面短测包装成创新成果。</p></header>
<nav><a href="#results">本轮实测</a><a href="#synthesis">机制与候选路线</a><a href="#papers">69篇覆盖</a><a href="执行合同.md">执行合同</a><a href="两分支可行性.csv">逐例表</a><a href="summary.json">汇总数据</a></nav>
<div class="notice">本轮没有重读69篇全文、没有新增独立验证，也没有更改场景。文献记录综合、标准数学恒等式和算法性能证据分别呈现。</div>
<section id="results">'''+render(result_text)+'''</section><section id="synthesis">'''+render(synthesis)+'''</section><section id="papers"><h2>69篇精读记录的机制覆盖</h2><p>全部纳入思路盘点，不等于全部能直接比较或每篇都有可落地组件。按编号展开查看原条件、方法、启示与正文定位；本轮新增迁移推导见上方六条路线。</p>'''+''.join(cards)+'''</section><footer>来源：本项目已保存精读记录与冻结开发IQ；AI辅助综合。未知量、未决项和失败结果均保留。没有使用另一模型模拟独立审稿。</footer></main></html>'''
    (HERE/'文献思想与候选方法.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(groups=groups, unresolved=sum(r['choice']=='unresolved' for r in rows), elapsed=done['seconds'], coverage=len(coverage)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
