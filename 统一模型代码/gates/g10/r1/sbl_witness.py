"""Native-evidence feasible-solution check; the candidate comes from blind LS."""
import torch
from ..physics import sbl_posterior
from .physics import Model
from .methods import run
from .config import BASE, DEFAULT
from .runtime import fresh, preflight, load_scene, write, read, sha, finish_audit
from .scenes import observation, digest


def main():
    inputs = [
        (BASE/'prepare_20261014_133623','3_0_0', '3_0_0_weak_SBL_clean.json',True),
        (BASE/'native_probe_20261014_140311','3_0_0','3_SBL.json',False),
    ]
    out=fresh('sbl_witness')
    preflight(out,dict(stage='SBL_FEASIBLE_NATIVE_EVIDENCE_CHECK',inference_uses_truth=False))
    evidence=[]
    for folder,key,filename,clean in inputs:
        r=read(folder/'rows'/filename)
        audit=read(folder/'final_audit_report.json')
        assert sha(folder/'rows'/filename)==audit['rows'][filename]
        scene=load_scene(folder,key)
        iq=scene['signal'][1].copy() if clean else observation(scene,'weak')[0]
        assert digest(iq)==r['input_sha']
        solved=run(iq,'LS',1,DEFAULT)
        m=Model(iq); noise=1/m.n
        native=[]
        for name,p in [('blind_LS',solved['positions']),('saved_SBL',r['result']['positions'])]:
            a=m.response(p)
            correlation=abs((a.mH@m.x[...,None]).squeeze(-1)).square().mean()
            gamma=(correlation-noise).clamp_min(1e-20).reshape(1)
            value=sbl_posterior(a,m.x,gamma,noise)[3].mean()
            native.append(dict(name=name,positions=p,gamma=float(gamma.cpu()),evidence=float(value.cpu())))
        evidence.append(dict(clean=clean,input_sha=digest(iq),source=str(folder/'rows'/filename),
                             full_dictionary_evidence=r['result']['events'][-1]['native_objective'],
                             feasible_one_atom=native,truth_used_only_for_display=r['truth']))
        del m
        torch.cuda.empty_cache()
    write(out/'report.json',dict(results=evidence,interpretation='A lower native objective is a feasible witness, not a global optimality proof.'))
    finish_audit(out)
    print(str(out)); print(evidence)


if __name__=='__main__':main()
