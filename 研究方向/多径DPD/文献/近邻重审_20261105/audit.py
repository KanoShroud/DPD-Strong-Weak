"""Check bibliography and additive Zotero collection writes without changing them."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def get(route):
    with urlopen('http://127.0.0.1:23119/api/users/0/' + route, timeout=30) as r:
        return json.load(r)


def main():
    before = {x['key']: x['data'] for x in json.loads((ROOT / 'sources/zotero_items.json').read_text(encoding='utf-8'))}
    rows = json.loads((ROOT / '文献矩阵.json').read_text(encoding='utf-8'))
    groups = {'A': 'FC7V9KDA', 'B': '47IRAJ5X', 'C': 'DEMZBKZG', 'D': 'LRSYGFUW'}
    checks, live = {}, []
    for grade, key in groups.items():
        members = [x for x in get(f'collections/{key}/items?limit=100') if not x['data'].get('parentItem')]
        expected = {k for r in rows if r['grade'] == grade for k in r['zotero_keys']}
        checks['collection_' + grade] = {x['key'] for x in members} == expected
        for item in members:
            d, old = item['data'], before[item['key']]
            checks['old_memberships_' + item['key']] = set(old.get('collections', [])) <= set(d.get('collections', []))
            for field in ['title', 'DOI', 'extra', 'creators']:
                checks[field + '_' + item['key']] = old.get(field) == d.get(field)
            checks['tags_' + item['key']] = sorted((x['tag'], x.get('type', 0)) for x in old.get('tags', [])) == sorted((x['tag'], x.get('type', 0)) for x in d.get('tags', []))
            live.append(item)
    checks['count_68'] = len(rows) == 68
    checks['unique_ids'] = len({r['id'] for r in rows}) == len(rows)
    checks['unique_library_items'] = len({k for r in rows for k in r['zotero_keys']}) == 37
    checks['pdfs_35'] = sum(r['local_pdf'] for r in rows) == 35
    checks['missing_33'] = sum(not r['local_pdf'] for r in rows) == 33
    manifest = json.loads((ROOT / 'sources/xia_manifest.json').read_text(encoding='utf-8'))
    checks['xia_pdf_unchanged'] = hashlib.sha256(Path(manifest['pdf']).read_bytes()).hexdigest() == manifest['sha256']
    checks['xia_structural_preflight'] = json.loads((ROOT / 'sources/xia_preflight.json').read_text(encoding='utf-8'))['verdict'] == 'PASS'
    result = dict(pass_all=all(checks.values()), checks=checks, historical_files_moved=False, localization_experiments_run=False,
                  scope='Metadata/collection and file identity checks; not full-paper semantic certification')
    (ROOT / 'sources/collection_items_after_v2.json').write_text(json.dumps(live, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'audit_v2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'pass_all': result['pass_all'], 'checks': len(checks), 'failed': [k for k, v in checks.items() if not v]}, ensure_ascii=False))
    assert result['pass_all']


if __name__ == '__main__':
    main()
