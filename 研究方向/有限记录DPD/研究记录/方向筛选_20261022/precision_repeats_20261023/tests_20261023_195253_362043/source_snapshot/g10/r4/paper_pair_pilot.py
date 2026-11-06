"""Nine-case AP5 alternating/joint paired timing pilot, not a formal replacement."""
from pathlib import Path
from datetime import datetime
import copy
import time
import numpy as np
import torch
from .paper_evidence import BASE, read, write, digest, metrics
from .formal_runtime import check, ensure_input
from .runtime import require_exclusive, resource_check
from .solvers import ap
from ..r3.solvers import Model, Trace, projected_gradient


def solve(iq, method, cfg):
    torch.cuda.synchronize()
    started = time.perf_counter()
    last_check = [started]

    def guard():
        now = time.perf_counter()
        if now-last_check[0] >= 1.:
            resource_check()
            last_check[0] = now

    trace = Trace(guard)
    model = Model(iq, guard, sample_rate_hz=cfg['sample_rate_hz'],
                  half_band_hz=cfg['half_band_hz'])
    trace.model = model
    detail = ap(model, trace, method, cfg)
    p = np.asarray(detail['output_positions'])
    value, gradient = model.joint(p, True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    result = dict(method=method, positions=p.tolist(), seconds=elapsed,
        joint_objective=value, projected_gradient=projected_gradient(p, gradient),
        counts=copy.deepcopy(model.counts), mechanism=detail)
    trace.model = None
    return result


def main():
    require_exclusive()
    resource_check()
    pointer = read(BASE/'formal_run.json')
    source = Path(pointer['out']).resolve(strict=True)
    assert digest((source/'contract.json').read_bytes()) == pointer['contract_sha']
    contract = check(source)
    assert read(source/'status.json')['status'] == 'COMPLETED'
    # Four classes, two endpoints, plus the already diagnosed difficult case.
    tasks = [(g, 400, gap, 'mixed') for g in (0, 5, 10, 15) for gap in (0, 15)]
    tasks.append((3, 400, 0, 'mixed'))
    data = []
    for task in tasks:
        name = '_'.join(map(str, task[:3]))
        assert (source/'inputs'/(name+'.json')).is_file()
        iq, truth, identity = ensure_input(source, task, contract)
        old = read(source/'rows'/(name+'_AP_5_JOINT.json'))
        audit = read(source/'final_audit_report.json')
        assert digest((source/'rows'/(name+'_AP_5_JOINT.json')).read_bytes()) == audit['rows'][name+'_AP_5_JOINT.json']
        assert identity == old['input_identity']
        data.append((name, iq, truth, identity, old))
    out = BASE/('paper_pair_pilot_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert not out.exists() and not out.is_relative_to(source)
    out.mkdir()
    cfg = data[0][-1]['result']['config']
    for item in data:
        assert item[-1]['result']['config'] == cfg
    write(out/'contract.json', dict(source=str(source), tasks=tasks, configuration=cfg,
        script_sha=digest(Path(__file__).read_bytes()), formal_contract_sha=pointer['contract_sha'],
        identity=[v[3] for v in data], note='Pilot only. Same native AP initialization, five branches and full-gradient stop.'))
    # Warm-up is a disposable run, never included in paired cost or evidence.
    start = time.perf_counter()
    solve(data[0][1], 'AP_5_JOINT', cfg)
    warmup = time.perf_counter()-start
    print(f'公共预热完成：{warmup:.2f}s（单列）', flush=True)
    pairs = []
    for index, (name, iq, truth, identity, old) in enumerate(data):
        methods = ('AP_5_ALT', 'AP_5_JOINT') if index % 2 == 0 else ('AP_5_JOINT', 'AP_5_ALT')
        pair = {}
        for method in methods:
            result = solve(iq, method, cfg)
            good, errors = metrics(truth, result['positions'])
            pair[method] = result
            write(out/(name+'_'+method+'.json'), dict(input_identity=identity, truth=truth.tolist(),
                result=result, both20=good, errors_m=errors.tolist()))
        a, b = pair['AP_5_ALT'], pair['AP_5_JOINT']
        assert np.allclose(a['mechanism']['candidate_list'], b['mechanism']['candidate_list'], rtol=0, atol=1e-8)
        assert np.allclose(b['mechanism']['candidate_list'], old['result']['mechanism']['candidate_list'], rtol=0, atol=1e-8)
        assert np.allclose(b['positions'], old['result']['positions'], rtol=0, atol=1e-5)
        branches = a['mechanism']['branches']+b['mechanism']['branches']
        assert all(v['stop_reason'] == 'COMMON_GRADIENT_REACHED' for v in branches)
        pairs.append(dict(case=name, alt_seconds=a['seconds'], joint_seconds=b['seconds'],
            alt_success=metrics(truth, a['positions'])[0], joint_success=metrics(truth, b['positions'])[0],
            alt_calls=a['counts']['joint_evaluations'], joint_calls=b['counts']['joint_evaluations']))
        print(f'五候选配对短测 {index+1}/{len(data)}：{name}，交替{a["seconds"]:.2f}s／联合{b["seconds"]:.2f}s', flush=True)
    alt = float(np.mean([p['alt_seconds'] for p in pairs]))
    joint = float(np.mean([p['joint_seconds'] for p in pairs]))
    # This is a coarse pilot estimate; 2400 cases, both methods, 25% margin + IO/QA.
    estimated = 2400*(alt+joint)*1.25+600
    check(source)
    write(out/'pilot_report.json', dict(passed=True, pairs=pairs, warmup_seconds=warmup,
        mean_alt_seconds=alt, mean_joint_seconds=joint, full_pair_estimated_seconds=estimated,
        estimate_note='9-case pilot; hard case deliberately represented; 25% margin plus 10 min IO/report. Not a bound.',
        next_status='FULL_PAIR_PENDING_TIME_APPROVAL', frozen_sources_unchanged=True))
    print(f'短测完成。完整4800次求解粗估{estimated/3600:.2f}小时，未启动。结果：{out}', flush=True)


if __name__ == '__main__':
    main()
