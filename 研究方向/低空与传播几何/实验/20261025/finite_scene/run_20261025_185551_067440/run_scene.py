"""Frozen small finite-scene diagnostic; no estimator or performance claims."""
import csv
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from finite_scene import fixed_panels, trace, unit_tests

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parents[1]/'research_admission_20261024/step1_deployment_20261025/run_20261025_142854_848860'
C = 299792458.


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def synthesize(links, coeff, freq, t):
    direct = np.zeros((2,5,len(t)),complex)
    H = np.zeros((2,5,len(freq)),complex)
    path_counts = []
    for q, source in enumerate(links):
        for m, link in enumerate(source):
            path_counts.append(len(link['paths']))
            for ray in link['paths']:
                length = ray['length_m']
                gain = (1. if ray['kind']=='direct' else -.5)/length
                direct[q,m] += gain*(coeff[q]@np.exp(2j*np.pi*freq[:,None]*(t[None,:]-length/C)))
                H[q,m] += gain*coeff[q]*np.exp(-2j*np.pi*freq*length/C)
    replay = H@np.exp(2j*np.pi*freq[:,None]*t[None,:])
    error = float(np.linalg.norm(direct-replay)/np.linalg.norm(direct))
    return direct, replay, H, error, path_counts


def main():
    begin = time.perf_counter()
    tests = unit_tests()
    if not all(tests.values()):
        raise RuntimeError(tests)
    out = ROOT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    assert out.resolve().is_relative_to(Path('F:/DPD_Strong_Weak/outputs_e2e').resolve())
    out.mkdir(exist_ok=False)
    paths = [ROOT/'finite_scene.py',Path(__file__),ROOT/'执行合同.md',PRIOR/'result.json',PRIOR/'iq_witness.npz']
    identity = {str(p.resolve(strict=True)):sha(p) for p in paths}
    (out/'preflight.json').write_text(json.dumps({'input_sha256':identity,'python':platform.python_version(),
            'numpy':np.__version__,'kind':'FINITE_SCENE_FORWARD_MODEL_DIAGNOSTIC'},ensure_ascii=False,indent=2),encoding='utf-8')
    for p in paths[:3]:
        (out/p.name).write_bytes(p.read_bytes())
    w = json.loads((PRIOR/'result.json').read_text(encoding='utf-8'))['witness']
    stations = np.array(w['stations_m'])
    positions = [np.array([x,y,4+h]) for x,y in [(80,60),(-120,140),(520,80),(-460,-120)] for h in [30,75,120]]
    positions += [np.array(w['unchanged_second_source_m']),np.array(w['alternate_position_m'])]
    panels = fixed_panels()
    scenes = {'empty': [],'single_A':panels[:1],'two_AB':panels}
    records,summary,iq_checks,compare = [],{}, {},{}
    with np.load(PRIOR/'iq_witness.npz') as old:
        coeff,freq,t = old['coefficients'],old['frequency_hz'],old['t_s']
    for name, faces in scenes.items():
        links = [[trace(p,s,faces) for s in stations] for p in positions]
        for i,(p,row) in enumerate(zip(positions,links)):
            for m,link in enumerate(row):
                reciprocal = trace(stations[m],p,faces)
                a,b = sorted(x['length_m'] for x in link['paths']),sorted(x['length_m'] for x in reciprocal['paths'])
                link['reciprocity_error_m'] = float(max(abs(np.array(a)-np.array(b)),default=0.)) if len(a)==len(b) else None
                records.append({'scene':name,'source_id':i,'station_id':m,'source_m':p.tolist(),**link})
        count = {'direct':0,'reflection_only':0,'empty':0,'direct_plus_reflection':0,'degenerate':0}
        los_per_source = []
        for row in links:
            los_per_source.append(sum(any(p['kind']=='direct' for p in r['paths']) for r in row))
            for link in row:
                direct = any(r['kind']=='direct' for r in link['paths'])
                refl = any(r['kind']=='reflection' for r in link['paths'])
                count['direct'] += int(direct)
                count['reflection_only'] += int(refl and not direct)
                count['empty'] += int(not link['paths'])
                count['direct_plus_reflection'] += int(direct and refl)
                count['degenerate'] += int(link['degenerate'])
        summary[name] = {**count,'links':70,'los_per_source':los_per_source}
        per_source,replay,H,error,path_counts = synthesize([links[9],links[12]],coeff,freq,t)
        iq_checks[name] = {'relative_replay_error':error,'path_counts_source_major':path_counts,
                          'source_A_los':los_per_source[9],'source_B_los':los_per_source[12]}
        np.savez_compressed(out/(name+'_iq.npz'),mixture=per_source.sum(axis=0),per_source=per_source,
             replay=replay,transfer=H,frequency_hz=freq,coefficients=coeff,t_s=t)
        lengths_a = [sorted(p['length_m'] for p in r['paths']) for r in links[9]]
        lengths_b = [sorted(p['length_m'] for p in r['paths']) for r in links[13]]
        same_count = all(len(a)==len(b) for a,b in zip(lengths_a,lengths_b))
        if same_count:
            shifts = np.concatenate([np.array(b)-np.array(a) for a,b in zip(lengths_a,lengths_b)])
            delta = float(np.mean(shifts))
            residual = float(np.max(abs(shifts-delta)))
        else:
            delta,residual = None,None
        compare[name] = {'same_environment_only':True,'original_counts':[len(x) for x in lengths_a],
                         'alternate_counts':[len(x) for x in lengths_b], 'counts_match':same_count,
                         'best_common_length_shift_m':delta,'max_length_shift_residual_m':residual,
                         'old_common_shift_equivalence_survives': bool(same_count and residual<1e-6)}
    checks = {**tests,'nondegenerate_frozen_links': all(not r['degenerate'] for r in records),
              'all_link_reciprocity': all(r['reciprocity_error_m'] is not None and r['reciprocity_error_m']<1e-7 for r in records),
              'all_specular_laws':all(p['reflection_law_error']<1e-10 and p['image_length_error_m']<1e-7 for r in records for p in r['paths'] if p['kind']=='reflection'),
              'all_iq_replays':all(c['relative_replay_error']<1e-10 for c in iq_checks.values()),
              'single_plane_blocked_has_no_specular_path':all(not r['paths'] for r in records if r['scene']=='single_A' and r['direct_blockers']),
              'input_identity_unchanged':all(sha(Path(p))==h for p,h in identity.items())}
    result = {'identity':identity,'stations_m':stations.tolist(),'positions_m':[p.tolist() for p in positions],
              'panels':[p.as_dict() for p in panels],'scene_panels':{k:[p.name for p in v] for k,v in scenes.items()},
              'summary':summary,'iq_checks':iq_checks,'fixed_environment_comparison':compare,
              'records':records,'checks':{k:bool(v) for k,v in checks.items()},
              'all_checks_pass':bool(all(checks.values())),'wall_seconds':time.perf_counter()-begin}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'path_list.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['scene','source_id','station_id','kind','panel','length_m','point_m'])
        writer.writeheader()
        for r in records:
            for p in r['paths']:
                writer.writerow({**{k:r[k] for k in ['scene','source_id','station_id']},
                                 **{k:p.get(k) for k in ['kind','panel','length_m','point_m']}})
    print(json.dumps({'out':str(out),'summary':summary,'iq':iq_checks,'comparison':compare,'checks':checks,'seconds':result['wall_seconds']},ensure_ascii=False))
    if not result['all_checks_pass']:
        raise RuntimeError('Frozen model diagnostic check failed; preserve outputs')


if __name__=='__main__':
    main()
