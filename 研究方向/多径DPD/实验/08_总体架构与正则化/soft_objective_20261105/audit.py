"""Final count and input identity audit; failed v1 stays a failed attempt."""
from soft_v2 import HERE,cfg,stable


def main():
    total=0
    for folder in ['math_v1','response_v1','math_v2','response_v2']:
        identity=cfg.read(HERE/folder/'identity.json');stable.inside.rr.check_hashes(identity);total+=len(identity)
    assert not cfg.read(HERE/'math_v1/summary.json')['passed']
    assert cfg.read(HERE/'math_v2/summary.json')['passed']
    rows=cfg.read(HERE/'response_v2/decisions.json')
    assert len(rows)==24 and sum(r['status']=='completed' for r in rows)==23
    a=cfg.read(HERE/'analysis.json')
    assert len(a['old_by_layout'])==24 and len(a['same_observation_initialization'])==2
    assert a['local_small_prediction_pass']==24
    assert cfg.read(HERE/'html_qa.json')['passed']
    cfg.write(HERE/'final_audit.json',dict(passed=True,hash_entries_checked=total,failed_attempt_preserved=True,
        completed_states=23,registered_states=24,no_new_scene=True,html_passed=True))
    print('PASS',total,'hash entries')


if __name__=='__main__':main()
