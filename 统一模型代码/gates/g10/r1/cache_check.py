"""Finite exact-cache equivalence/timing probe; never launches formal tasks."""
import gc
import time
from pathlib import Path
import numpy as np
import torch
from .config import BASE
from .runtime import fresh, preflight, read, write, sha, load_scene, Ledger, finish_audit
from .scenes import observation, digest
from .physics import Model
from .sbl import statistics
from .sbl_outputs import load_state
from .runner import record, warmup


def main():
    prior = Path(read(BASE/'release.json')['preparation']).resolve(strict=True)
    audit = read(prior/'final_audit_report.json')
    assert audit['identity_passed']
    out = fresh('cache_check')
    preflight(out, dict(stage='exact_cache_probe', prior=str(prior),
                        changed_only='dictionary_cache', formal_launched=False))
    print('OUT='+str(out), flush=True)
    clock = Ledger(out)
    warmup()
    tag = '0_30_15_strong_SBL'
    reference = read(prior/'rows'/f'{tag}.json')
    assert sha(prior/'rows'/f'{tag}.json') == audit['rows'][tag+'.json']
    scene = load_scene(prior, '0_30_15')
    iq, _, _ = observation(scene, 'strong')
    assert digest(iq) == reference['input_sha']
    manifest = read(prior/'states'/tag/'manifest.json')
    arrays, _ = load_state(prior/'states'/tag, manifest['states'][manifest['roles']['25']])
    m = Model(iq, clock.check)
    m.sbl_spectral_weight = torch.tensor(arrays['spectral_weight'], device=m.device)
    gamma = torch.tensor(arrays['gamma'], device=m.device)
    times, results = {}, {}
    torch.cuda.reset_peak_memory_stats()
    for enabled in (False, True):
        if enabled:
            torch.cuda.synchronize(); start = time.perf_counter()
            assert m.enable_response_cache(arrays['points'], 128), 'Cache does not fit existing safety limit'
            torch.cuda.synchronize()
            build = time.perf_counter()-start
        results[enabled] = statistics(m, arrays['points'], gamma, 128, True)
        samples = []
        for _ in range(3):
            torch.cuda.synchronize(); start = time.perf_counter()
            statistics(m, arrays['points'], gamma, 128, True)
            torch.cuda.synchronize(); samples.append(time.perf_counter()-start)
        times[str(enabled)] = samples
    errors = {}
    for key in ('em', 'power', 'numerator', 'denominator'):
        a, b = [results[enabled][key].cpu().numpy() for enabled in (False, True)]
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
        errors[key] = float(np.max(abs(a-b)))
    np.testing.assert_allclose(results[False]['evidence'], results[True]['evidence'], rtol=1e-12)
    micro = dict(passed=True, seconds=times, build_seconds=build, max_absolute_errors=errors,
                 speedup=float(np.median(times['False'])/np.median(times['True'])),
                 cache_bytes=m.counts['dictionary_cache_bytes'],
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write(out/'micro.json', micro)
    del m, results, gamma
    gc.collect(); torch.cuda.empty_cache()
    print(f"逐遍核验通过，缓存耗时比={1/micro['speedup']:.3f}；开始有限完整轨迹核验。", flush=True)
    if micro['speedup'] <= 1.05:
        write(out/'report.json',dict(status='NO_USEFUL_CACHE_GAIN',micro=micro))
        finish_audit(out)
        return
    comparisons = []
    for task in ((0,30,15,'strong'), (0,30,0,'weak'), (2,30,15,'mixed'), (3,30,0,'weak')):
        tag = f'{task[0]}_{task[1]}_{task[2]}_{task[3]}_SBL'
        assert sha(prior/'rows'/f'{tag}.json') == audit['rows'][tag+'.json']
        old = read(prior/'rows'/f'{tag}.json')
        new = record(out, clock, task, 'SBL', dict(old['result']['config'], dictionary_cache=True))
        assert old['input_sha'] == new['input_sha']
        a,b = old['result'],new['result']
        assert a['stop_reason']==b['stop_reason'] and a['counts']['iterations']==b['counts']['iterations']
        np.testing.assert_allclose([h['native_objective'] for h in a['history']],
                                   [h['native_objective'] for h in b['history']], rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose([h['positions'] for h in a['history']],
                                   [h['positions'] for h in b['history']],rtol=0,atol=1e-6)
        np.testing.assert_allclose(a['positions'],b['positions'],rtol=0,atol=1e-6)
        ar,br=[read(root/'sbl_outputs'/f'{tag}.json') for root in (prior,out)]
        for role in ar['roles']:
            aa,bb=ar['states'][ar['roles'][role]],br['states'][br['roles'][role]]
            assert aa['iteration']==bb['iteration']
            for variant in aa['outputs']:
                np.testing.assert_allclose(aa['outputs'][variant],bb['outputs'][variant],rtol=0,atol=1e-6)
        comparisons.append(dict(tag=tag,kind=task[3],passed=True,
            old_seconds=a['seconds'],new_seconds=b['seconds'],ratio=b['seconds']/a['seconds'],
            iterations=b['counts']['iterations'],stop=b['stop_reason'],
            cache_bytes=b['counts'].get('dictionary_cache_bytes'),
            history_position_max_difference=float(np.max(abs(np.asarray([h['positions'] for h in a['history']])-
                                                                np.asarray([h['positions'] for h in b['history']]))))))
        write(out/'report.json',dict(status='IN_PROGRESS',micro=micro,comparisons=comparisons,prior=str(prior)))
        print(f"{len(comparisons)}/4 {tag}: {a['seconds']:.1f}s → {b['seconds']:.1f}s，轨迹核验通过",flush=True)
    clock.flush()
    finish_audit(out)
    write(out/'report.json',dict(status='PASSED',micro=micro,comparisons=comparisons,prior=str(prior),
                               seconds=clock.state['seconds']))


if __name__ == '__main__':
    main()
