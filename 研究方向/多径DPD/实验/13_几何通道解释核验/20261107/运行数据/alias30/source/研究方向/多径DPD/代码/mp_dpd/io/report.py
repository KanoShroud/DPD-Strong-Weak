# ruff: noqa: F401
# Preserve imported globals for legacy function cloning and reexports.
"""Saved-result summaries; no method selection from truth."""
from pathlib import Path
import html
import numpy as np
import 研究方向.多径DPD.代码.mp_dpd.models.scene as cfg


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
            normal_mean_seconds=float(np.mean([r['seconds'] for r in rr if r['normal']])) if any(r['normal'] for r in rr) else None,
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
    # Pilot IN/OUT/MIX weights 1/1/2 differ from formal 4/4/4.
    forecast_components = None
    if phase == 'pilot' and len(rows) == 52:
        forecast_components = dict(generation=0., MP=0., FLOOD=0., XIA=0.)
        for condition in cfg.CONDITIONS:
            for kind in ('IN', 'OUT', 'MIX'):
                group = [r for r in rows if r['condition'] == condition['name'] and r['kind'] == kind]
                assert group
                forecast_components['generation'] += 80*float(np.mean([r['generation_seconds'] for r in group]))
                for method in ('MP', 'FLOOD', 'XIA'):
                    forecast_components[method] += 80*float(np.mean([r['methods'][method]['seconds'] for r in group]))
    forecast = sum(forecast_components.values()) if forecast_components else None
    notes = []
    if rows and any(overall[m]['normal'] < .9*len(rows) for m in ('LOS', 'MP')):
        notes.append('LOS or MP numerical normal rate below 90%; review required before formal release')
    value = dict(phase=phase, completed=len(rows), expected=expected, finished=finished,
                 metrics=overall, conditions=groups, layouts=by_layout,
                 generation_plus_algorithm_seconds=total, forecast_formal_seconds=forecast,
                 forecast_components_seconds=forecast_components,
                 forecast_rule='condition x geometry stratum mean x 80; failed algorithm timings retained',
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
        lines += ['', f'按信道条件×构型类别分层外推正式生成与算法时间：{forecast/3600:.2f}小时；不包含报告、磁盘审计，不能保证新构型尾部成本。',
                  '每个条件×构型类别的短测均值乘80（4构型×20重复），再求和；另留30%余量是工程预留，不是置信上界。',
                  '当前异常返回也计入成本；修复失败后必须重新估时，不能将异常快速返回当成功加速。']
    lines += ['', '## 放行提示', '', *(notes or ['无当前数值集中失效提示；仍须正确性审计和源码身份核验通过。'])]
    (out/'结果报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    escaped = html.escape('\n'.join(lines))
    (out/'结果报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>四站物理场景比较</title><style>body{max-width:1100px;margin:40px auto;padding:24px;background:#f5f6f2;color:#18383d;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}pre{white-space:pre-wrap;background:white;padding:28px;border-radius:16px;overflow-wrap:anywhere}a{color:#127578}</style><h1>四站物理场景比较</h1><p><a href="summary.json">结构化结果</a> · <a href="结果报告.md">完整报告</a></p><pre>'+escaped+'</pre></html>', encoding='utf-8')
    return value
