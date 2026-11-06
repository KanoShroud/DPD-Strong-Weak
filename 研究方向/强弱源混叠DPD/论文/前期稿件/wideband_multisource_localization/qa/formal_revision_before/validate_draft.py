"""Lightweight manuscript QA; small algebra checks, no localization experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent


def math_checks():
    rng = np.random.default_rng(20261017)
    reports = []
    for trial in range(8):
        f = np.linspace(-6e6, 6e6, 5)
        stations = np.array([[500, 0], [0, 500], [-500, 0], [0, -500]])
        z = rng.uniform(-300, 300, 2)
        y = rng.normal(size=(5, 4)) + 1j*rng.normal(size=(5, 4))
        energy = float(np.sum(abs(y)**2))

        def response(p):
            delta = p-stations
            d = np.linalg.norm(delta, axis=1)
            v = delta/d[:, None]
            k = 2*np.pi*f[:, None]/3e8
            a = np.exp(-1j*k*d)/d
            t = -1/d-1j*k
            loggrad = t[:, :, None]*v[None]
            vv = v[:, :, None]*v[:, None, :]
            logh = vv[None]/d[None, :, None, None]**2
            logh = logh+t[:, :, None, None]/d[None, :, None, None]*(np.eye(2)-vv)[None]
            da = a[:, :, None]*loggrad
            d2a = a[:, :, None, None]*(logh+loggrad[:, :, :, None]*loggrad[:, :, None, :])
            return a, da, d2a

        def profile(p):
            a, da, _ = response(p)
            q = np.sum(a.conj()*y, axis=1)/np.sum(abs(a)**2, axis=1)
            e = y-a*q[:, None]
            value = np.sum(abs(e)**2)/energy
            gradient = -2*np.real(np.einsum('fl,fli,f->i', e.conj(), da, q))/energy
            return value, gradient

        a, da, d2a = response(z)
        q = np.sum(a.conj()*y, axis=1)/np.sum(abs(a)**2, axis=1)
        e = y-a*q[:, None]
        fixed = np.zeros((2, 2)); correction = np.zeros((2, 2))
        for fi in range(len(f)):
            b = da[fi]*q[fi]
            fixed += 2*np.real(b.conj().T@b-np.einsum('l,lij->ij', e[fi].conj(), d2a[fi]*q[fi]))/energy
            cross = np.empty((2, 2))
            for nu, t in enumerate((1, 1j)):
                cross[:, nu] = 2*np.real(np.einsum('l,li->i', (a[fi]*t).conj(), b)
                                        -np.einsum('l,li->i', e[fi].conj(), da[fi]*t))/energy
            huu = 2*np.vdot(a[fi], a[fi]).real/energy
            correction += cross@cross.T/huu
        vp = fixed-correction
        h = .002
        eye = np.eye(2)
        fdg = np.array([(profile(z+h*v)[0]-profile(z-h*v)[0])/(2*h) for v in eye])
        fdh = np.column_stack([(profile(z+h*v)[1]-profile(z-h*v)[1])/(2*h) for v in eye])
        eg = float(np.max(abs(fdg-profile(z)[1])))
        eh = float(np.max(abs(fdh-vp)))
        # Independent projection identity for arbitrary complex B and candidate.
        b = rng.normal(size=(4, 1))+1j*rng.normal(size=(4, 1))
        atom = rng.normal(size=4)+1j*rng.normal(size=4)
        obs = y[0]
        pb = np.eye(4)-b@np.linalg.pinv(b)
        u, r = pb@atom, pb@obs
        gain = abs(np.vdot(u, r))**2/np.vdot(u, u).real
        both = np.column_stack([b, atom])
        residual2 = obs-both@np.linalg.lstsq(both, obs, rcond=None)[0]
        ep = float(abs(np.vdot(r, r).real-np.vdot(residual2, residual2).real-gain))
        psd = float(np.linalg.eigvalsh(correction).min())
        if eg > 1e-7 or eh > 1e-7 or ep > 1e-10 or psd < -1e-12:
            raise AssertionError((trial, eg, eh, ep, psd))
        reports.append(dict(trial=trial, gradient_abs_error=eg, hessian_abs_error=eh,
                            projection_abs_error=ep, schur_min_eigenvalue=psd))
    # Local rate statement under its explicitly positive-definite assumptions.
    correction = np.array([[9., 0.], [0., 1.]])
    hvp = np.array([[1., .1], [.1, 2.]])
    hfixed = hvp+correction
    rate = np.linalg.eigvals(np.eye(2)-np.linalg.solve(hfixed, hvp))
    assert np.all(rate >= 0) and np.all(rate < 1)
    return dict(passed=True, trials=reports, local_rate_eigenvalues=rate.tolist(),
                scope='Synthetic small-matrix algebra/finite differences only; not localization performance.')


def protected_tokens():
    result = {}
    for path in sorted((HERE / 'sections').glob('*.tex')):
        if path.name.startswith('generated_'):
            continue  # Mechanical tables are not part of the prose-only edit.
        txt = path.read_text(encoding='utf-8')
        blocks = re.findall(r'\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}', txt, flags=re.S)
        inline = re.findall(r'(?<!\\)\$[^$]*?(?<!\\)\$', txt, flags=re.S)
        refs = re.findall(r'\\(?:cite|label|ref|eqref)\{[^}]+\}', txt)
        numbers = re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', txt)
        payload = json.dumps(dict(blocks=blocks, inline=inline, refs=refs, numbers=numbers), ensure_ascii=False)
        result[path.name] = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-polish', action='store_true')
    parser.add_argument('--check-polish', action='store_true')
    args = parser.parse_args()
    (HERE / 'qa').mkdir(exist_ok=True)
    checks = math_checks()
    texts = [p.read_text(encoding='utf-8') for p in sorted((HERE / 'sections').glob('*.tex'))]
    text = '\n'.join(texts)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    references = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', text)
    cites = set(','.join(re.findall(r'\\cite\{([^}]+)\}', text)).split(','))-{''}
    bib = (HERE / 'references.bib').read_text(encoding='utf-8')
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
    if len(labels) != len(set(labels)):
        raise AssertionError('Duplicate labels')
    if set(references)-set(labels) or cites-bibkeys:
        raise AssertionError(dict(missing_labels=list(set(references)-set(labels)), missing_citations=list(cites-bibkeys)))
    forbidden = ('F:\\', 'outputs_e2e', 'G10-', 'record_sha', 'checkpoint', '通过门禁')
    if any(x in text for x in forbidden):
        raise AssertionError('Internal project terms leaked into manuscript')
    for name in ('geometry', 'power_gap', 'paired_cost', 'candidate_case', 'error_cdf', 'curvature_cost'):
        for suffix in ('pdf', 'png', 'csv'):
            if not (HERE / 'figures' / f'{name}.{suffix}').is_file():
                raise AssertionError('Missing figure format')
    checks.update(labels=len(labels), cross_references=len(references), citations=len(cites),
                  latex_sections=len(texts), forbidden_terms_checked=list(forbidden))
    token_path = HERE / 'qa/pre_polish_tokens.json'
    if args.snapshot_polish:
        token_path.write_text(json.dumps(protected_tokens(), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.check_polish:
        before = {k: v for k, v in json.loads(token_path.read_text(encoding='utf-8')).items()
                  if not k.startswith('generated_')}
        if before != protected_tokens():
            raise AssertionError('Math/numbers/citations changed during language-only polish')
        checks['polish_protected_tokens_unchanged'] = True
    log = HERE / 'main.log'
    if log.exists():
        content = log.read_text(encoding='utf-8', errors='replace')
        errors = re.findall(r'^!.*|^.*(?:undefined references|Citation .* undefined|Reference .* undefined|Missing character:|Overfull \\hbox).*', content, re.M)
        checks['latex_errors'] = errors
        if errors:
            raise AssertionError(errors)
    (HERE / 'qa/validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in checks.items() if k != 'trials'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
