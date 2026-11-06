"""Bounded new-rate integration check, not the 320-row qualification or formal run."""
import json
import numpy as np
from .config import DEFAULT, METHODS, FS, N
from .runtime import fresh, preflight, write, Ledger, finish_audit, sha
from .runner import record, scene_for, warmup
from .test_r1 import suite
from .output_check import duplicate_identity_test


def main():
    out = fresh('sampling_short')
    # Unseen realizations for this bounded check; formal repeats begin at 100.
    tasks = [((2, 20, 0, 'weak'), method) for method in METHODS]
    tasks += [((0, 21, 15, 'mixed'), method) for method in ('LS', 'AP', 'SBL')]
    cfg = dict(DEFAULT)
    preflight(out, dict(stage='SAMPLING_INTEGRATION_SHORT', tasks=tasks, config=cfg,
                        sample_rate_hz=FS,N=N,no_formal_release=True,
                        success100_required=False, no_per_case_wall_cap=True))
    print('OUT='+str(out), flush=True)
    tests = suite(); tests['duplicate_identity'] = duplicate_identity_test()
    write(out/'numerical_tests.json', tests)
    if not tests['passed']:
        raise RuntimeError('Numerical test failed')
    write(out/'warmup.json',dict(seconds=warmup(), separate=True))
    clock = Ledger(out); rows = []
    for i, (task, method) in enumerate(tasks):
        print(f'{i+1}/{len(tasks)} {method} {task}', flush=True)
        r = record(out, clock, task, method, cfg)
        rows.append(dict(tag=r['tag'], method=method, kind=r['kind'], errors_m=r['metrics']['errors_m'],
                         seconds=r['result']['seconds'], stop=r['result']['stop_reason'],
                         iterations=r['result']['counts']['iterations'], input_sha=r['input_sha']))
        write(out/'summary.json',dict(rows=rows,formal_ready=False))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    task, method = tasks[0]
    path = out/'rows'/f'{rows[0]["tag"]}.json'
    before = sha(path)
    record(out, clock, task, method, cfg)
    resume_ok = before == sha(path)
    scene = scene_for(out, 2, 20, 0)
    freq = np.fft.fftfreq(N, 1/FS); mask = abs(freq) <= scene['meta']['bandwidth_hz']/2
    signal, noise = np.fft.fft(scene['signal'][1], axis=-1)/N, np.fft.fft(scene['noise'], axis=-1)/N
    snr = 10*np.log10(np.mean(abs(signal[:, mask])**2)/np.mean(abs(noise[:, mask])**2))
    write(out/'summary.json',dict(rows=rows,formal_ready=False,numerical_passed=True,resume_ok=resume_ok,
          sample_rate_hz=FS,N=N,duration_seconds=N/FS,measured_mean_station_inband_snr_db=float(snr),
          rationale='Engineering and finite native runs, no ranking or sufficiency release'))
    clock.flush(); finish_audit(out)
    print('DONE '+str(out), flush=True)


if __name__ == '__main__':
    main()
