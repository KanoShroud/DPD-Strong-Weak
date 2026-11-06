"""Read saved evidence by condition; no new scenario or method selection."""
import numpy as np
from soft_v2 import HERE,BASE,PREV,INPUT,SAVED,cfg,stable


def rms(errors):return float(np.sqrt(np.mean(np.square(errors))))


def main():
    hashes=cfg.read(HERE/'response_v2/identity.json');stable.inside.rr.check_hashes(hashes)
    rows=[]
    for d in cfg.read(PREV/'discrepancy_v1/blind_decisions.json'):
        tag=d['tag'];old=cfg.read(SAVED/(tag+'_ORTH_SAFE.json'))['result']
        with np.load(INPUT/'banks'/(str(d['layout'])+'_00.npz'),allow_pickle=False) as z:truth=z['p']
        rows.append(dict(tag=tag,condition=d['condition'],layout=d['layout'],
            los_errors=stable.inside.rr.evaluation(np.asarray(old['los']['p']),truth),
            mp_errors=stable.inside.rr.evaluation(np.asarray(old['final']['p']),truth),
            mp_normal=old['normal'],mp_K=len(old['paths']),mp_seconds=old['seconds']))
    groups=[]
    for condition in ['L0','L1_P075','L2_P075','L3_P150']:
        rr=[r for r in rows if r['condition']==condition]
        case=[rms(r['mp_errors']) for r in rr]
        groups.append(dict(condition=condition,count=len(rr),los_rmse=rms([r['los_errors'] for r in rr]),
            mp_rmse=rms([r['mp_errors'] for r in rr]),mp_median_case_rmse=float(np.median(case)),
            mp_max_case_rmse=max(case),normal=sum(r['mp_normal'] for r in rr),estimated_K=[r['mp_K'] for r in rr]))
    paired=cfg.read(BASE/'source_path_gate_20261105/run_v1/summary.json')['rows'];comparison=[]
    for condition in ['L2_P075','L3_P150']:
        rr=[r for r in paired if r['tag'].endswith(condition)]
        comparison.append(dict(condition=condition,count=len(rr),
            geometry_rmse=rms([r['GEOMETRY_FIRST']['errors'] for r in rr]),
            near_rmse=rms([r['DIAGNOSTIC_NEAREST']['errors'] for r in rr]),
            geometry_normal=sum(r['GEOMETRY_FIRST']['normal'] for r in rr),
            near_normal=sum(r['DIAGNOSTIC_NEAREST']['normal'] for r in rr),
            near_max_case_rmse=max(r['DIAGNOSTIC_NEAREST']['rmse'] for r in rr)))
    decisions=cfg.read(HERE/'response_v2/decisions.json');evaluation=cfg.read(HERE/'response_v2/evaluation.json');soft=[]
    for arm in ['BLIND','DIAGNOSTIC']:
        rr=[r for r in evaluation if r['arm']==arm and r['status']=='completed']
        for i,factor in enumerate([.005,.01,1.]):
            soft.append(dict(arm=arm,factor=factor,count=len(rr),reference_rmse=rms([r['reference_errors'] for r in rr]),
                updated_rmse=rms([r['trials'][i]['errors'] for r in rr]),
                improved=sum(rms(r['trials'][i]['errors'])<rms(r['reference_errors']) for r in rr),
                stationary=sum(r['trials'][i]['stationary'] for r in rr)))
    local=[r for r in decisions if r.get('reference',{}).get('stationary',False) and r.get('sensitivity',{}).get('valid',False)]
    small=[t['fit'] for r in local for t in r['trials'] if t['factor']<1]
    result=dict(old_all_conditions=groups,old_by_layout=rows,same_observation_initialization=comparison,
        soft_control=soft,local_valid_states=len(local),local_small_checks=len(small),
        local_small_prediction_pass=sum(t['small_response_pass'] for t in small),
        local_small_stationary=sum(t['stationary'] for t in small),
        response_max_seconds=max(r.get('seconds',0) for r in decisions),
        response_all_seconds=cfg.read(HERE/'response_v2/summary.json')['seconds'],
        identity_unchanged=True,no_new_scene=True,no_full_blind_algorithm=True)
    cfg.write(HERE/'analysis.json',result);print(result)


if __name__=='__main__':main()
