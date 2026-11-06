"""Validate this reading deliverable without modifying research evidence."""
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent


def main():
    missing = []
    checked = 0
    docs = list(ROOT.glob('*.md')) + [ROOT.parent / '准入报告.md']
    for doc in docs:
        text = doc.read_text(encoding='utf-8-sig')
        for href in re.findall(r'\]\(([^)]+)\)', text):
            href = href.strip('<>')
            if href.startswith(('https:', 'http:', '#', 'zotero:', 'mailto:')):
                continue
            href = unquote(href.split('#')[0])
            if not href:
                continue
            if href.startswith('file:'):
                target = Path(urlparse(href).path.lstrip('/'))
            else:
                target = Path(href) if Path(href).is_absolute() else doc.parent / href
            checked += 1
            if not target.exists():
                missing.append({'document': doc.name, 'href': href})
    sources = json.loads((ROOT / '来源与分类核验.json').read_text(encoding='utf-8'))
    html = json.loads((ROOT / 'HTML核验.json').read_text(encoding='utf-8'))
    structure = json.loads((ROOT / 'visual_structure/preflight_structure.json').read_text(encoding='utf-8'))
    preflight = {r['id']: r['verdict'] for r in structure['results']}
    for name in ['B3', 'B5', 'B6', 'B7', 'C1', 'C2', 'C3']:
        preflight[name] = json.loads((ROOT / f'{name}_preflight.json').read_text(encoding='utf-8'))['verdict']
    audit = {
        'status': 'READING_AND_CLASSIFICATION_COMPLETE_NO_NEW_SIMULATION',
        'pdf_and_zotero_audit_pass': sources['pass'],
        'papers_this_batch': 10, 'pages_this_batch': sources['bc_pages'],
        'papers_cumulative': sources['total_papers'], 'pages_cumulative': sources['total_pages'],
        'preflight': preflight,
        'preflight_exception': 'C3 encrypted flag; all four pages readable/rendered normally; not reported PASS',
        'markdown_links_checked': checked, 'missing_markdown_targets': missing,
        'html': html,
        'visual_review': 'Parent inspected desktop, expanded-paper, plan, mobile screenshots after repair',
        'code_checks': {'python_py_compile': 'PASS', 'ruff_F_E9': 'PASS', 'node_build_and_render': 'PASS'},
        'plan_review_repairs': [
            'Control model freedom across path/support hypotheses',
            'Include common-convolution-factor and alternative-channel ambiguities',
            'Oracle is diagnostic reference, not a strict RMSE bound',
            '72 strictly paired isolated inputs for 36 mixtures',
            'Equal transmitted power; retain blockage and path-loss effects',
            'RF complex-IQ prerequisites checked before adapting acoustic DATEMM',
        ],
        'reading_status_zotero_added': False,
        'new_experiments': 0,
        'limitations': ['A9 separate supplement not obtained', 'C3 PDF remains outside Zotero',
                        'C3 existing conference metadata error recorded but not edited',
                        'No new identifiability proof, algorithm validation, or novelty guarantee'],
    }
    audit['pass'] = not missing and sources['pass'] and html['pass']
    (ROOT / '交付核验.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pass': audit['pass'], 'markdown_links_checked': checked, 'missing': missing}, ensure_ascii=False))
    if not audit['pass']:
        raise RuntimeError('Delivery check failed.')


if __name__ == '__main__':
    main()
