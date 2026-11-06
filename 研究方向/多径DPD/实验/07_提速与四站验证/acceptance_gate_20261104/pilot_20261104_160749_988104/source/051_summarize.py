"""Saved-result analysis only; no estimator, no selective omission."""
import argparse
from pathlib import Path
import json
import hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parent/'risk_calibration_20261102/formal_20261102_181736_622970'


def load(p):
    data = json.loads(p.read_text(encoding='utf-8'))
    if 'payload' in data and 'sha256' in data:
        encoded = json.dumps(data['payload'], ensure_ascii=False, allow_nan=False, sort_keys=True).encode('utf-8')
        assert hashlib.sha256(encoded).hexdigest() == data['sha256']
        return data['payload']
    return data


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('folder'); args = ap.parse_args()
    folder = Path(args.folder).resolve(strict=True)
    data = load(folder/'results.json')
    assert data['complete']
    rows = data['rows']; summary = []
    methods = sorted({r['method'] for r in rows})
    candidate = 'HYBRID' if 'HYBRID' in methods else 'SQN'
    for c in ['L0', 'L2', 'L3', 'HIGH']:
        for method in methods:
            a = [r for r in rows if r['case'].startswith(c+'_') and r['method'] == method]
            times = np.array([r['seconds'] for r in a])
            old = [load(FORMAL/'cases'/r['case']/'solve.json') for r in a]
            # Truth appears only in this offline analysis, never estimator.
            contract = load(FORMAL/'contract.json')['contract']
            with np.load(contract['conditions'][c]['bank'], allow_pickle=False) as z:
                truth = z['p']
            def mse(p):
                return min(np.sum((p-truth)**2), np.sum((p-truth[::-1])**2))/2
            record = dict(condition=c, method=method, n=len(a),
                rmse_m=float(np.sqrt(np.mean([r['rmse_m']**2 for r in a]))),
                original_rmse_m=float(np.sqrt(np.mean([mse(np.array(d['final']['p'])) for d in old]))),
                seconds_mean=float(times.mean()), seconds_p95=float(np.quantile(times, .95)), seconds_max=float(times.max()),
                within3=sum(r['normal'] and r['seconds'] <= 3 for r in a),
                within5=sum(r['normal'] and r['seconds'] <= 5 for r in a),
                natural=sum(r['normal'] for r in a),
                mean_K=float(np.mean([r['K'] for r in a])),
                frontend_mean=float(np.mean([r['front_seconds'] for r in a])),
                search_mean=float(np.mean([r['search_seconds'] for r in a])),
                fit_mean=float(np.mean([r['fit_seconds'] for r in a])))
            summary.append(record)
    output = dict(source=str(folder), complete=True, rows=len(rows), summary=summary,
        all_audits=all(r['audit'] for r in rows), identity=load(folder/'postflight.json'),
        failures=[r for r in rows if not r['normal']],
        exceeds5=[r for r in rows if r['method'] == candidate and r['seconds'] > 5])
    (folder/'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 自然停止配对复核', '', '## Material Passport', '',
        '- Origin Skill: experiment-agent', '- Origin Mode: run', '- Origin Date: 2026-11-04',
        '- Verification Status: VERIFIED (saved original-formula audits only; not a publication or generalization verdict)',
        '- Version Label: natural_stop_pair_v1', '',
        '完整两区域初始化、全频点、全时延搜索、成组增径及原梯度停止门槛。GN重算局部曲率；SQN复用缩放正定曲率并解小型凸二次步；HYBRID在内部点使用QN，触界或线搜索停滞时转完整目标SLSQP。以目录identity登记的具体版本为准。时间从IQ到返回，不以时钟停止。', '',
        '|条件|后端|匹配RMSE/m|平均/s|P95/s|最大/s|自然结束|≤3s且自然结束|≤5s且自然结束|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['condition']}|{r['method']}|{r['rmse_m']:.6f}|{r['seconds_mean']:.3f}|{r['seconds_p95']:.3f}|{r['seconds_max']:.3f}|{r['natural']}/{r['n']}|{r['within3']}/{r['n']}|{r['within5']}/{r['n']}|")
    lines += ['', '## 与原逐条充分多径解的精度比较', '',
        '这里只比较同一IQ的定位误差。旧结果计时来自不同批次，且前端及增径组织不同，不用旧计时归因本轮的单一改动。', '',
        f'|条件|原充分解RMSE/m|当前{candidate} RMSE/m|差值/m|', '|---|---:|---:|---:|']
    for r in summary:
        if r['method'] == candidate:
            lines.append(f"|{r['condition']}|{r['original_rmse_m']:.6f}|{r['rmse_m']:.6f}|{r['rmse_m']-r['original_rmse_m']:+.6f}|")
    lines += ['', f'## 成本构成（{candidate}）', '', '|条件|前端均值/s|搜索均值/s|精修均值/s|平均路径数|', '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['method'] == candidate:
            lines.append(f"|{r['condition']}|{r['frontend_mean']:.3f}|{r['search_mean']:.3f}|{r['fit_mean']:.3f}|{r['mean_K']:.1f}|")
    lines += ['', '## 边界与未收敛结果', '',
        '旧28条、每条件7次噪声重复，不是多构型/独立信道试验；L3_R01曾触发数值修正，明确属于开发暴露。全部未收敛输出保留误差和耗时，不剔除。HIGH是已有四反射压力条件，并未加入新场景。', '',
        f"原公式目标、PG和可行性审计：{output['all_audits']}；运行前后身份：{output['identity']['unchanged']}。", '',
        '|案例|后端|停止原因|PG|耗时/s|RMSE/m|', '|---|---|---|---:|---:|---:|']
    for r in output['failures']:
        lines.append(f"|{r['case']}|{r['method']}|{r['stop']}|{r['PG']:.3g}|{r['seconds']:.3f}|{r['rmse_m']:.6f}|")
    (folder/'配对结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
