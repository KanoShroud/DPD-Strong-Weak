"""Simulation-only clean-center linearization, not an available estimator."""
from datetime import datetime
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import audit as a


def main():
    start = time.perf_counter()
    contract = a.load(a.FORMAL/'contract.json')['contract']
    a.cal.verify(contract)
    out = a.HERE/('clean_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    files = [Path(__file__), a.HERE/'audit.py', a.HERE/'执行协议.md']
    hashes = {str(p): a.ref.sha(p) for p in files}
    a.write(out/'preflight.json', dict(hashes=hashes, contract=contract,
        role='oracle_clean_diagnostic_no_new_blind_results', seed=2026110208))
    (out/'source').mkdir()
    for p in files:
        (out/'source'/p.name).write_bytes(p.read_bytes())
    folder = a.FORMAL/'cases/L0_R00'
    _, clean, bank, _ = a.case_data(folder, contract)
    solved = a.load(folder/'solve.json')
    f, y, n, _ = a.cal.spectral(clean)
    x, fit = a.ref.los_fit(np.array(solved['los']['x']), f, y, n)
    variance = float(bank['variance'])
    diag, l = a.risk.diagnose(x, f, y, [], n, variance, np.random.default_rng(2026110208))
    result = dict(fit=fit, diagnostic=diag, risk_selector_modified=False)
    if l is not None:
        np.savez_compressed(out/'influence.npz', L=l, x=x)
        truth = bank['p']
        p = x[:4].reshape(2,2)*a.ref.UNIT
        if a.norm2(p-truth[::-1]) < a.norm2(p-truth):
            truth = truth[::-1]
        bias = (p-truth).ravel()
        rows = []
        for folder in sorted((a.FORMAL/'cases').glob('L0_R*')):
            iq, sameclean, _, info = a.case_data(folder, contract)
            assert np.array_equal(clean, sameclean)
            _, noise_f, _, _ = a.cal.spectral(iq-clean)
            delta = l@a.ref.real(noise_f)
            estimate = np.array(a.load(folder/'solve.json')['los']['p'])
            if a.norm2(estimate-p[::-1]) < a.norm2(estimate-p):
                estimate = estimate[::-1]
            observed = (estimate-p).ravel()
            error = (estimate-truth).ravel()
            rows.append(dict(case=info['case']['id'], predicted_response=delta,
                observed_response=observed, response_gap=np.linalg.norm(delta-observed),
                actual_loss=a.norm2(error)/2, predicted_loss=a.norm2(bias+delta)/2,
                linear_noise_loss=a.norm2(delta)/2))
        covariance = variance*(l@l.T)
        ev, vectors = np.linalg.eigh(covariance)
        assert ev[0] > 0
        rng = np.random.default_rng(2026110208)
        # Only 4-D fixed linear estimator responses, not new RF IQ or localization fits.
        draws = rng.normal(size=(100000, 8, 4))@(vectors*np.sqrt(ev)).T
        mc_loss = np.sum((draws+bias)**2, axis=2).mean(axis=1)/2
        actual = float(np.mean([d['actual_loss'] for d in rows]))
        pred = float(np.mean([d['predicted_loss'] for d in rows]))
        result.update(rows=rows, clean_bias_m=bias, covariance_m2=covariance,
            expected_linear_mse=(np.trace(covariance)+a.norm2(bias))/2,
            observed_mse=actual, realized_linear_mse=pred,
            fixed_linear_eight_sample_interval95=np.quantile(mc_loss,[.025,.975]),
            empirical_percentile_actual=float(np.mean(mc_loss<=actual)),
            empirical_percentile_linear=float(np.mean(mc_loss<=pred)),
            inference='conditional_fixed_linear_distribution_not_unconditional_calibration_proof')
    a.cal.verify(contract)
    assert all(a.ref.sha(p)==h for p,h in hashes.items())
    result['seconds'] = time.perf_counter()-start
    a.write(out/'result.json', result)
    a.write(a.HERE/'latest_clean.json', dict(path=str(out)))
    print('结果目录',out,flush=True)
    print({k:v for k,v in result.items() if k not in ('fit','diagnostic','rows','covariance_m2')},flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
