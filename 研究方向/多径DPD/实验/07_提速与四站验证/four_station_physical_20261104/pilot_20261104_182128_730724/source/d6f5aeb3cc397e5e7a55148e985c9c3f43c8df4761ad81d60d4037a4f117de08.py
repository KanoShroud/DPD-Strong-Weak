"""Saved-result summaries; no method selection from truth."""
from pathlib import Path
import html
import numpy as np
import scene as cfg


METHODS = ('GEO', 'LOS', 'MP', 'FLOOD', 'XIA')


def metrics(rows):
    out = {}
    for method in METHODS:
        rr = [r['methods'][method] for r in rows if method in r['methods']]
        full = [r for r in rr if r['errors_m'] is not None]
        times = [r['seconds'] for r in rr]
        value = float(np.sqrt(np.mean([np.asarray(r['errors_m'])**2 for r in full]))) if full else None
        out[method] = dict(n=len(rr), complete=len(full), normal=sum(r['normal'] for r in rr),
            full_rmse_m=value if len(full) == len(rr) else None, conditional_rmse_m=value,
            mean_seconds=float(np.mean(times)) if times else None,
            p95_seconds=float(np.percentile(times, 95)) if times else None,
            max_seconds=max(times) if times else None,
            below3=sum(t <= 3 for t in times), below5=sum(t <= 5 for t in times))
    return out


def summarize(out, phase, finished=False):
    out = Path(out)
    rows = [cfg.read(p) for p in sorted((out/'cases').glob('*/evaluation.json'))]
    overall = metrics(rows)
    groups = {c['name']: metrics([r for r in rows if r['condition'] == c['name']]) for c in cfg.CONDITIONS}
    by_layout = {str(k): metrics([r for r in rows if r['layout'] == k]) for k in sorted({r['layout'] for r in rows})}
    expected = 52 if phase == 'pilot' else 3120
    total = sum(r['generation_seconds']+r['methods']['MP']['seconds']+
                r['methods']['FLOOD']['seconds']+r['methods']['XIA']['seconds'] for r in rows)
    # Each pilot condition has four layouts; formal has 12*20 realizations.
    forecast = total*60 if phase == 'pilot' and len(rows) == 52 else None
    notes = []
    if rows and any(overall[m]['normal'] < .9*len(rows) for m in ('LOS', 'MP')):
        notes.append('LOS or MP numerical normal rate below 90%; review required before formal release')
    value = dict(phase=phase, completed=len(rows), expected=expected, finished=finished,
                 metrics=overall, conditions=groups, layouts=by_layout,
                 generation_plus_algorithm_seconds=total, forecast_formal_seconds=forecast,
                 review_flags=notes, runtime_excludes_report_audit_io=True)
    cfg.write(out/'summary.json', value)
    def fmt(x):
        return '—' if x is None else f'{x:.3f}'
    lines = ['# 四站物理场景比较', '', f'完成 {len(rows)}/{expected} 条。阶段：{phase}。', '',
             'RMSE单位m；完整耗时单位s。缺失输出只给条件RMSE并列分母，非驻定有限结果保留。GEO/LOS为同一DPD流程的阶段输出，不能重复计入运行总账。', '',
             '|方法|完整输出|数值/流程正常|全体RMSE|完整子集RMSE|平均秒|P95秒|最大秒|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m, d in overall.items():
        lines.append(f"|{m}|{d['complete']}/{d['n']}|{d['normal']}/{d['n']}|{fmt(d['full_rmse_m'])}|{fmt(d['conditional_rmse_m'])}|{fmt(d['mean_seconds'])}|{fmt(d['p95_seconds'])}|{fmt(d['max_seconds'])}|")
    lines += ['', '## 分条件RMSE', '', '|条件|GEO|LOS|MP|Flood|Xia|', '|---|---:|---:|---:|---:|---:|']
    for c, ms in groups.items():
        lines.append('|'+c+'|'+'|'.join(fmt(ms[m]['full_rmse_m']) for m in METHODS)+'|')
    if forecast is not None:
        lines += ['', f'按52条条件均值线性外推正式生成与算法时间：{forecast/3600:.2f}小时；不包含报告、磁盘审计，不能保证新构型尾部成本。',
                  '正式点估计是各条件成本乘240，建议另留30%调度/长尾余量；不是置信上界。']
    lines += ['', '## 放行提示', '', *(notes or ['无当前数值集中失效提示；仍须正确性审计和源码身份核验通过。'])]
    (out/'结果报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    escaped = html.escape('\n'.join(lines))
    (out/'结果报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>四站物理场景比较</title><style>body{max-width:1100px;margin:40px auto;padding:24px;background:#f5f6f2;color:#18383d;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}pre{white-space:pre-wrap;background:white;padding:28px;border-radius:16px;overflow-wrap:anywhere}a{color:#127578}</style><h1>四站物理场景比较</h1><p><a href="summary.json">结构化结果</a> · <a href="结果报告.md">完整报告</a></p><pre>'+escaped+'</pre></html>', encoding='utf-8')
    return value
