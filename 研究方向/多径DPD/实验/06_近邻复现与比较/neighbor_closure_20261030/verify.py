"""Audit claims/counts/identities; never re-evaluates localization performance."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

from collect import OUT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = {}
    previous = json.loads((OUT.parent/'formula_audit_20261030/source_manifest.json').read_text(encoding='utf-8'))
    for path, h in previous['identities'].items():
        if Path(path).suffix == '.py':
            checks['frozen/'+Path(path).name] = digest(path) == h
    a = json.loads((OUT/'search_raw.json').read_text(encoding='utf-8'))
    b = json.loads((OUT/'search_extended.json').read_text(encoding='utf-8'))
    seen, total, passed = set(), 0, 0
    for i in range(12):
        r = a['results']['crossref_'+str(i)]
        if r.get('status') != 200:
            r = b['results']['retry_'+str(i)]
        passed += r.get('status') == 200
        for w in r.get('data', {}).get('message', {}).get('items', []):
            total += 1
            seen.add(w.get('DOI','').lower() or w.get('title',[''])[0])
    for key, r in b['results'].items():
        if key.startswith('extra_'):
            passed += r.get('status') == 200
            for w in r.get('data', {}).get('message', {}).get('items', []):
                total += 1
                seen.add(w.get('DOI','').lower() or w.get('title',[''])[0])
    citing = []
    for key, r in a['results'].items():
        if key.startswith('citing_'):
            citing.extend(r.get('data', {}).get('results', []))
    for r in json.loads((OUT/'snowball_extended.json').read_text(encoding='utf-8')).values():
        citing.extend(r.get('data', {}).get('results', []))
    rows = json.loads((OUT/'neighbor_matrix.json').read_text(encoding='utf-8'))
    missing = sum(not r['zotero_key'] for r in rows)
    checks['counts'] = (passed,total,len(seen),len(citing),len({w['id'] for w in citing}),len(rows),missing)==(17,525,461,122,117,46,16)
    checks['unique_matrix_ids'] = len({r['id'] for r in rows})==len(rows)
    checks['all_sources_labeled'] = all(r['evidence'] and r['implication'] for r in rows)
    ident = json.loads((OUT/'source_identity.json').read_text(encoding='utf-8'))
    checks['Friedlander_PDF_identity'] = digest(ident['pdf']) == ident['pdf_sha256']
    algebra = json.loads((OUT/'geometry_identity.json').read_text(encoding='utf-8'))
    checks['algebra_100_not_localization'] = algebra['passed'] and len(algebra['rows'])==100 and algebra['worst_reflection_relative_difference']<1e-11
    for f in OUT.glob('*.md'):
        s = f.read_text(encoding='utf-8')
        checks['utf8/'+f.name] = '\ufffd' not in s
        for link in re.findall(r'\]\(([^)]+)\)',s):
            if not link.startswith(('http:', 'https:', 'zotero:', '#')):
                target=(f.parent/unquote(link.split('#')[0])).resolve()
                checks['link/'+f.name+'/'+link]=target.exists()
    for f in ['当前任务.md','对话上下文.md','研究总结.md','实验结果记录.md','修改记录.md','统一端到端模型研究.md']:
        s=(Path(r'F:\DPD_Strong_Weak')/f).read_text(encoding='utf-8')
        checks['root_utf8/'+f]='\ufffd' not in s
        checks['root_current_link/'+f]='neighbor_closure_20261030' in s
    outputs = {str(f.relative_to(OUT)): digest(f) for f in OUT.iterdir() if f.is_file() and f.name!='review_audit.json'}
    result=dict(kind='LITERATURE_AND_ALGEBRA_AUDIT_NO_NEW_LOCALIZATION',
                passed=all(checks.values()), checks=checks, evidence_sha256=outputs,
                query_count=passed, returned_records=total, unique_query_records=len(seen),
                forward_records=len(citing), unique_forward_records=len({w['id'] for w in citing}),
                matrix_items=len(rows), missing_zotero=missing)
    (OUT/'review_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['passed'], {k:v for k,v in checks.items() if not v}
    print('PASS',len(checks),'checks; no changed frozen solver')


if __name__=='__main__':
    main()
