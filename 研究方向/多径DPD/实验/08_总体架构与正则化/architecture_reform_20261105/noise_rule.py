"""Analytical noise-event and shrinkage diagnostic; not a deployed lambda rule."""
from pathlib import Path
import numpy as np
from scipy.stats import gamma
from threadpoolctl import threadpool_limits
from math_checks import cfg, ad, stable, HERE, INPUT


def main():
    assert cfg.read(HERE/'run_v2/summary.json')['mathematical_gate']
    stable.inside.rr.check_hashes(cfg.read(HERE/'run_v2/identity.json'))
    out = HERE/'noise_rule_v1'
    out.mkdir(exist_ok=False)
    hashes = {str(p): cfg.sha(p) for p in [Path(__file__).resolve(), HERE/'noise_rule_protocol.md', HERE/'math_checks.py']}
    cfg.write(out/'identity.json', hashes)
    keep = abs(np.fft.fftfreq(cfg.N, 1/cfg.FS))<=cfg.B/2
    nf = int(keep.sum()); alpha = .05
    sigma = np.sqrt(cfg.NOISE_VARIANCE)
    universal = sigma*np.sqrt(gamma.isf(alpha/4, nf))
    fixed = sigma*np.sqrt(np.log(8*(cfg.N-1)/alpha))
    rows = []
    for i in range(100, 106):
        with np.load(INPUT/'banks'/f'{i}_00.npz', allow_pickle=False) as z:
            bank = {k: z[k] for k in z.files}
        s = np.column_stack([ad.wave.synthesize(bank['bits'][q], int(bank['first']), bank['offsets'][q], np.arange(cfg.N)/cfg.FS)/np.sqrt(bank['pulse_norm2']) for q in range(2)])
        norm = np.linalg.norm(np.fft.fft(s, axis=0, norm='ortho')[keep], axis=0)
        noise = np.fft.fft(bank['noise'], axis=-1, norm='ortho')[:, keep]
        for name in ['L2_P075', 'L3_P150']:
            condition = next(c for c in cfg.CONDITIONS if c['name']==name)
            _, power = cfg.observation(bank, condition)
            strength = abs(bank['gains'][:, :, None])*np.sqrt(power)*norm[None, :, None]
            for m in range(4):
                for q in range(2):
                    for ell in condition['ids']:
                        ratio = float(universal/strength[m, q, ell])
                        rows.append(dict(layout=i, condition=name, station=m, source=q, path=ell,
                            component_norm=float(strength[m, q, ell]), universal_over_component=ratio,
                            isolated_noiseless_retained_fraction=max(0., 1-ratio),
                            observed_noise_norm=float(np.linalg.norm(noise[m])),
                            oracle_noise_diagnostic_only=True))
    cfg.write(out/'rows.json', rows)
    summary = dict(frequencies=nf, alpha=alpha, sigma_oracle=sigma, universal_lambda=universal,
        fixed_dictionary_lambda=fixed, universal_over_fixed=universal/fixed,
        components=len(rows), fully_shrunk_isolated=sum(r['universal_over_component']>=1 for r in rows),
        median_retained_isolated=float(np.median([r['isolated_noiseless_retained_fraction'] for r in rows])),
        maximum_noise_over_bound=max(r['observed_noise_norm']/universal for r in rows),
        localization_tested=False, lambda_approved_for_deployment=False)
    cfg.write(out/'summary.json', summary)
    stable.inside.rr.check_hashes(hashes)
    print(summary, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
