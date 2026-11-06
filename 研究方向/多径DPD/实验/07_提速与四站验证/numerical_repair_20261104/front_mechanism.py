"""Posthoc source-contribution audit; no new localization rule or truth access by solvers."""
import numpy as np
from threadpoolctl import threadpool_limits
import stable

cfg, ad, HERE = stable.cfg, stable.ad, stable.HERE
INPUT = HERE.parent/'inside_pilot_20261104/batch_v2'


def coherence(engine, y, f, n, points):
    waves = [engine.engine.rank_one(y, f, p)[1] for p in points]
    product = waves[0].conj()*waves[1]
    bins = np.rint(f/cfg.FS*n).astype(int) % n
    spectrum = np.zeros(n, complex); spectrum[bins] = product
    denom = np.linalg.norm(waves[0])*np.linalg.norm(waves[1])
    return float(np.max(abs(np.fft.ifft(spectrum)))*n/denom)


def main():
    assert cfg.read(HERE/'paired_v1/summary.json')['numerical_gate_passed']
    stable.inside.rr.check_hashes(cfg.read(INPUT/'identity.json'))
    e = stable.inside.Engine()
    rows = []
    for path in sorted((INPUT/'cases').iterdir()):
        for name, digest in cfg.read(path/'receipt.json').items():
            assert cfg.sha(path/name) == digest
        ev = cfg.read(path/'evaluation.json')
        with np.load(path/'iq.npz', allow_pickle=False) as z:
            iq = z['iq']
        front = e.frontend(iq)
        chosen = np.array(e.geo_snapshot['selected'])
        f, y, n, _ = ad.b.fc.cal.spectral(iq)
        # All above calculations are blind. Truth/bank opened only for attribution below.
        with np.load(INPUT/'banks'/f"{ev['layout']:02d}_00.npz", allow_pickle=False) as z:
            bank = dict(z)
        truth = bank['p']
        distances = np.linalg.norm(front['refined'][:, None]-truth[None], axis=2)
        oracle_ids = distances.argmin(axis=0)
        selected = front['refined'][chosen]
        condition = next(c for c in cfg.CONDITIONS if c['name'] == ev['condition'])
        rebuilt, powers = cfg.observation(bank, condition)
        assert cfg.digest(rebuilt) == cfg.digest(iq)
        components = bank['los']+np.sum(bank['reflected']*np.sqrt(powers.transpose(1, 0, 2))[..., None], axis=2)
        assert np.allclose(components.sum(axis=0)+bank['noise'], iq, rtol=1e-13, atol=1e-20)
        length = 2**int(np.ceil(np.log2(2*n-1)))
        freq = np.fft.fftfreq(length, 1/cfg.FS)
        mask = abs(freq) <= cfg.B/2
        spectra = np.fft.fft(components, n=length, axis=2)[:, :, mask]
        allspec = np.fft.fft(iq, n=length, axis=1)[:, mask]
        ratios = []
        for k, (m, st) in enumerate(e.pairs):
            phase = np.exp(2j*np.pi*freq[mask, None]*(front['lags'][k]/cfg.C)[None])
            corr_a = (spectra[0, st]*spectra[0, m].conj())@phase/length/n
            corr_b = (spectra[1, st]*spectra[1, m].conj())@phase/length/n
            total = (allspec[st]*allspec[m].conj())@phase/length/n
            other = total-corr_a-corr_b
            magnitude = np.stack([abs(corr_a), abs(corr_b), abs(other)], axis=1)
            ratios.append(magnitude/magnitude.sum(axis=1, keepdims=True))
        fractions = np.concatenate(ratios, axis=0)
        def describe(ids):
            w = front['reward'][ids]
            attribution = (w@fractions)/np.maximum(w.sum(axis=1, keepdims=True), 1e-15)
            return dict(points=front['refined'][ids], total_reward=float(np.maximum(w[0], w[1]).sum()),
                        weighted_absolute_contribution=attribution,
                        rank_one_waveform_max_coherence=coherence(e, y, f, n, front['refined'][ids]))
        rows.append(dict(tag=ev['tag'], geo_errors=stable.inside.rr.evaluation(selected, truth),
            nearest_candidate_m=distances.min(axis=0), selected=describe(chosen),
            nearest_truth_pair=describe(oracle_ids), distinct_oracle_candidates=bool(oracle_ids[0] != oracle_ids[1]),
            warning='Nearest-truth pair and source decomposition are diagnostic only; contribution magnitudes do not add as signed power'))
    cfg.write(HERE/'front_mechanism.json', dict(rows=rows, source_sha=cfg.sha(__file__),
        scope='No learned threshold, no changed score, no new localization output; seen R00 only'))
    for r in rows:
        err = np.sqrt(np.mean(np.square(r['geo_errors'])))
        if err > 20:
            print(r['tag'],round(err,2),'coverage',np.round(r['nearest_candidate_m'],2),
                  'score',round(r['selected']['total_reward'],2),round(r['nearest_truth_pair']['total_reward'],2),
                  'coherence',round(r['selected']['rank_one_waveform_max_coherence'],3),
                  round(r['nearest_truth_pair']['rank_one_waveform_max_coherence'],3),
                  'fractions',np.round(r['selected']['weighted_absolute_contribution'],2))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
