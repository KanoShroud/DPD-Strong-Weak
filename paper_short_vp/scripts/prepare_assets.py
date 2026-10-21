"""Rebuild short-paper figures and audit saved results; never run a locator."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('FC_VP', 'VP_VP')
STYLE = {'FC_VP': ('#3C5488', '--', 'FC--VP'),
         'VP_VP': ('#E64B35', '-', 'VP--VP')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def import_evidence(project):
    src = project / 'paper/data/review_supplement_20261020'
    audit = json.loads((src / 'audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'COMPLETED_OFFLINE_AUDIT'
    manifest = dict(protocol=audit['protocol'], source_files={},
                    literature_sources=[], paired=audit['paired'],
                    configurations=audit['frozen_configs'],
                    geometry_seed=audit['geometry_seed'],
                    waveform_seed=audit['waveform_seed'])
    for name in ('per_case.csv', 'summary.csv'):
        path = src / name
        assert sha(path) == sha(ROOT / 'data' / name)
        manifest['source_files'][str(path)] = sha(path)
    for path in (src / 'audit.json',
                 project / '统一模型代码/gates/g10/r4/adaptation_review.py',
                 project / '统一模型代码/gates/g10/r4/bnomp.py',
                 project / '统一模型代码/gates/g10/r4/paper_checks/review_supplement.py'):
        manifest['source_files'][str(path)] = sha(path)
    # The long manuscript is maintained concurrently and may absorb its supplement.
    # It is a writing reference, not a numerical dependency of this short paper.
    manifest['writing_references'] = {
        str(path): sha(path) for path in (
            project / 'paper/paper_submission.tex', project / 'paper/paper_supplement.tex')
        if path.exists()}
    # Verify the source audit's individual identities without executing solvers.
    checked = 0
    for field, content in audit.items():
        if isinstance(content, dict):
            for path, expected in content.items():
                if isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected):
                    p = Path(path)
                    if p.is_absolute() and p.exists():
                        assert sha(p) == expected, str(p)
                        checked += 1
    manifest['source_identity_checks'] = checked
    layouts = []
    raw = project / 'outputs_e2e/unified/e2e_g10/r4/formal_recovered_20261017_093236_136920'
    for i in range(20):
        metadata_path = raw / 'inputs' / f'{i}_400_0.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        manifest['source_files'][str(metadata_path)] = sha(metadata_path)
        for k, position in enumerate(metadata['positions']):
            layouts.append(dict(stage='mechanism', layout=i, source=k,
                                x_m=position[0], y_m=position[1]))
    for i, entry in enumerate(audit['layouts']):
        for k, position in enumerate(entry['positions']):
            layouts.append(dict(stage='geometry', layout=i, source=k,
                                x_m=position[0], y_m=position[1]))
    write_csv(ROOT / 'data/source_positions.csv', layouts)
    manifest['realizations'] = dict(mechanism=[400, 410], geometry=list(range(1000, 1005)))
    manifest['mechanism_seeds'] = dict(geometry=20261017041, waveform=20261017042)
    manifest['template_hashes'] = {}
    for name in ('cas-sc.cls', 'cas-common.sty', 'cas-model2-names.bst'):
        p = project / 'paper/template' / name
        assert sha(p) == sha(ROOT / 'template' / name)
        manifest['template_hashes'][name] = sha(p)
    # Subset the already verified BibTeX without copying unrelated references.
    selected = ('ChenHudsonYao2002', 'WeissAmar2005', 'Mamandipoor2016',
                'Yang2022', 'Zeng2025', 'GolubPereyra1973', 'ITURP5252024')
    bib = (project / 'paper/references/references.bib').read_text(encoding='utf-8')
    entries = re.split(r'(?m)(?=^@)', bib)
    chosen = [entry.strip() for entry in entries
              if any(re.match(r'@\w+\{' + key + ',', entry) for key in selected)]
    assert len(chosen) == len(selected)
    (ROOT / 'references/references.bib').write_text('\n\n'.join(chosen)+'\n', encoding='utf-8')
    manifest['literature_sources'] = [
        {'key': key, 'source': 'paper/references/references.bib'} for key in selected]
    manifest['public_checks'] = [
        'https://arxiv.org/html/1509.01942v2#S2.SS1',
        'https://doi.org/10.1016/j.jsv.2021.116549',
        'https://doi.org/10.1109/ACCESS.2025.3645128',
        'https://doi.org/10.1137/0710036']
    (ROOT / 'data/source_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def check_documents():
    import fitz
    en = (ROOT / 'main.tex').read_text(encoding='utf-8')
    zh = (ROOT / 'main_zh.tex').read_text(encoding='utf-8')
    blocks = r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}'
    normalize = lambda text: re.sub(r'\s+', '', text)
    equations = [[normalize(body) for _, body in re.findall(blocks, text, re.S)]
                 for text in (en, zh)]
    assert equations[0] == equations[1] and len(equations[0]) == 7
    get_cites = lambda text: re.findall(r'\\cite\{([^}]+)\}', text)
    assert get_cites(en) == get_cites(zh)
    citations = set(','.join(get_cites(en)).split(','))
    bib = (ROOT / 'references/references.bib').read_text(encoding='utf-8')
    assert citations <= set(re.findall(r'@\w+\{([^,]+),', bib))
    result_lines = [[line.strip() for line in text.splitlines()
                     if re.match(r'^(FC--VP|VP--VP|AP1|AP5|VP-BNOMP) &', line)]
                    for text in (en, zh)]
    assert result_lines[0] == result_lines[1] and len(result_lines[0]) == 5
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', en, re.S).group(1)
    abstract_words = len(abstract.split())
    assert abstract_words <= 250
    manifest = json.loads((ROOT / 'data/source_manifest.json').read_text(encoding='utf-8'))
    for name, value in manifest['template_hashes'].items():
        assert sha(ROOT / 'template' / name) == value
    report = dict(bilingual_equation_blocks=7, bilingual_result_rows=5,
                  same_citations=True, abstract_words=abstract_words,
                  template_files_unchanged=True, documents={}, figures={})
    for name in ('main', 'main_zh'):
        tex = (ROOT / (name+'.tex')).read_text(encoding='utf-8')
        labels = re.findall(r'\\label\{([^}]+)\}', tex)
        refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex)
        assert len(labels) == len(set(labels)) and set(refs) <= set(labels)
        assert '\\documentclass[a4paper,12pt,review]{cas-sc}' in tex
        assert not any(s in tex for s in ('margin=', '@fleqnfalse', '\\cs_set', '\\resizebox'))
        log = (ROOT / 'build' / (name+'.log')).read_text(encoding='utf-8', errors='replace')
        assert 'Missing character' not in log
        assert not re.search(r'(Reference|Citation).*undefined|There were undefined', log)
        warnings = re.findall(r'Overfull.*', log)
        # CAS 2.4 reports an internal front-matter box wider than its vbox.
        # It occurs before body text; page-coordinate checks and rendering
        # independently test whether any printed content is actually clipped.
        assert all('129.88684pt' in warning for warning in warnings), warnings
        document = fitz.open(ROOT / (name+'.pdf'))
        if name == 'main':
            assert len(document) <= 10
        outside = []
        for i, page in enumerate(document):
            assert abs(page.rect.width-595.28) < .1 and abs(page.rect.height-841.89) < .1
            text = page.get_text()
            assert f'Page {i+1} of {len(document)}' in text
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        box = fitz.Rect(span['bbox'])
                        if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width+1 or box.y1 > page.rect.height+1:
                            outside.append((i+1, span['text']))
        assert not outside, outside
        report['documents'][name] = dict(pages=len(document), a4=True, labels=len(labels),
            sha256=sha(ROOT / (name+'.pdf')), source_sha256=sha(ROOT / (name+'.tex')),
            page_bounds_ok=True, template_frontmatter_box_warnings=warnings)
    for name in ('error_cdf', 'cost_cdf'):
        doc = fitz.open(ROOT / 'figures' / (name+'.pdf'))
        assert sum(len(p.get_images()) for p in doc) == 0
        fonts = doc[0].get_fonts()
        assert fonts and all(len(doc.extract_font(f[0])[3]) > 0 for f in fonts)
        report['figures'][name] = dict(vector=True, embedded_fonts=True)
    report['passed'] = True
    (ROOT / 'build/document_check.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-from', type=Path)
    parser.add_argument('--check-documents', action='store_true')
    args = parser.parse_args()
    if args.check_documents:
        check_documents()
        return
    if args.import_from:
        import_evidence(args.import_from)
    rows = read_csv(ROOT / 'data/per_case.csv')
    saved = read_csv(ROOT / 'data/summary.csv')
    assert len(rows) == 1920
    calculated = []
    for s in saved:
        rr = [r for r in rows if (r['stage'], r['method']) == (s['stage'], s['method'])]
        assert len(rr) == int(s['n'])
        strong = np.array([float(r['strong_error_m']) for r in rr])
        weak = np.array([float(r['weak_error_m']) for r in rr])
        times = np.array([float(r['seconds']) for r in rr])
        assert np.isfinite(np.concatenate((strong, weak, times))).all()
        both = (strong <= 20.) & (weak <= 20.)
        assert int(both.sum()) == int(s['both_count'])
        for key, value in (
            ('strong_rmse_m', np.sqrt(np.mean(strong**2))),
            ('weak_rmse_m', np.sqrt(np.mean(weak**2))),
            ('mean_seconds', times.mean()),
            ('median_seconds', np.median(times)),
            ('p95_seconds', np.quantile(times, .95))):
            np.testing.assert_allclose(value, float(s[key]), rtol=1e-12, atol=1e-12)
        calculated.append(s)
    paired = {m: {r['case']: r for r in rows if r['stage'] == 'mechanism' and r['method'] == m}
              for m in METHODS}
    assert len(paired[METHODS[0]]) == len(paired[METHODS[1]]) == 240
    assert paired[METHODS[0]].keys() == paired[METHODS[1]].keys()
    assert all(paired[METHODS[0]][k]['both20'] == paired[METHODS[1]][k]['both20']
               for k in paired[METHODS[0]])
    fc = np.array([float(r['seconds']) for r in paired['FC_VP'].values()])
    vp = np.array([float(paired['VP_VP'][k]['seconds']) for k in paired['FC_VP']])
    saving = 100 * (1-vp.mean()/fc.mean())
    assert round(saving, 2) == 83.77
    assert int((vp < fc).sum()) == 170
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'legend.fontsize': 8.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    exported = []
    for name, fields, labels in (
        ('error_cdf', ('strong_error_m', 'weak_error_m'),
         ('Strong-source error (m)', 'Weak-source error (m)')),
        ('cost_cdf', ('seconds', 'cycles'), ('Complete runtime (s)', 'Cyclic sweeps'))):
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.35), layout='constrained')
        plotrows = []
        for j, (ax, field, label) in enumerate(zip(axes, fields, labels)):
            for method in METHODS:
                values = np.sort([float(r[field]) for r in paired[method].values()])
                prob = np.arange(1, len(values)+1)/len(values)
                color, line, title = STYLE[method]
                ax.step(values, prob, where='post', color=color, ls=line, lw=1.7,
                        label=title)
                plotrows.extend(dict(panel=j+1, method=method, value=float(x), cdf=float(y))
                                for x, y in zip(values, prob))
            ax.set_xscale('log')
            ax.set_ylim(0, 1.025)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xlabel(label)
            ax.set_ylabel('Empirical CDF')
            ax.grid(alpha=.16, which='major')
            ax.text(.03, .95, f'({chr(97+j)})', transform=ax.transAxes,
                    va='top', fontweight='bold')
            ax.legend(loc='lower right', frameon=False)
            if name == 'error_cdf':
                ax.axvline(20, color='#555555', lw=.8, ls=':')
                ax.set_xlim(.003 if j == 0 else .01, 30 if j == 0 else 1000)
        fig.savefig(ROOT / 'figures' / (name+'.pdf'), bbox_inches='tight')
        fig.savefig(ROOT / 'figures' / (name+'.png'), dpi=600, bbox_inches='tight')
        plt.close(fig)
        write_csv(ROOT / 'figures' / (name+'.csv'), plotrows)
        exported.append(name)
    report = dict(passed=True, rows=len(rows), mechanism_pairs=240,
                  new_geometry_observations=480, same_success_identity=True,
                  mean_saving_percent=float(saving), vp_faster=170,
                  figures=exported, evaluated_from='saved positions and complete runtime',
                  no_locator_run=True)
    (ROOT / 'build/data_check.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
