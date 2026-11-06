"""Read-only verification of approved Zotero changes and source identities."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API = 'http://127.0.0.1:23119/api/users/0'
GROUPS = {
    '4LXLXLLD': ['B6', 'B7'],
    'U9ZXZSL9': ['A9', 'B3', 'B5', 'A6'],
    'HUJQVA9D': ['A2', 'A3', 'A4', 'C3'],
    'WTFNSPGV': ['A1', 'A2', 'A7', 'A8', 'B1', 'B2', 'B4'],
    '9ZAUQ8GQ': ['A5', 'A6', 'A8', 'C3'],
    'P3R7ZER6': ['C1', 'C2'],
}


def get(route):
    response = requests.get(API + route, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text(encoding='utf-8'))
    before = json.loads((ROOT / 'zotero_items_before.json').read_text(encoding='utf-8'))
    audit = {'checked_at': datetime.now(timezone.utc).isoformat(), 'items': [], 'sources': []}
    for ref in manifest:
        for item in ref['items']:
            key = item['key']
            now = get('/items/' + key)['data']
            old = before[key]
            changes = [field for field in set(old) | set(now)
                       if field not in {'collections', 'version', 'dateModified'}
                       and old.get(field) != now.get(field)]
            expected = ['YX7G7ZKP'] + [k for k, ids in GROUPS.items() if ref['id'] in ids]
            audit['items'].append({
                'id': ref['id'], 'key': key,
                'unexpected_field_changes': changes,
                'tags_unchanged': old.get('tags') == now.get('tags'),
                'extra_unchanged': old.get('extra') == now.get('extra'),
                'old_memberships_preserved': set(old.get('collections', [])) <= set(now['collections']),
                'expected_memberships_present': set(expected) <= set(now['collections']),
                'new_memberships': sorted(set(now['collections']) - set(old.get('collections', []))),
            })
            for att in item['attachments']:
                actual = hashlib.sha256(Path(att['path']).read_bytes()).hexdigest()
                audit['sources'].append({'id': ref['id'], 'pages': att['pages'],
                                         'sha256_matches': actual == att['sha256']})
    c3 = json.loads((ROOT / 'C3_source.json').read_text(encoding='utf-8'))
    audit['sources'].append({'id': 'C3', 'pages': c3['pages'], 'source': 'author-hosted PDF',
                             'sha256_matches': hashlib.sha256(Path(c3['path']).read_bytes()).hexdigest() == c3['sha256']})
    audit['categories'] = [get('/collections/' + k)['data'] for k in ['YX7G7ZKP', *GROUPS]]
    audit['total_papers'] = len(audit['sources'])
    audit['total_pages'] = sum(x['pages'] for x in audit['sources'])
    audit['bc_pages'] = sum(x['pages'] for x in audit['sources'] if x['id'][0] in 'BC')
    audit['pass'] = all(
        not x['unexpected_field_changes'] and x['tags_unchanged'] and x['extra_unchanged']
        and x['old_memberships_preserved'] and x['expected_memberships_present']
        for x in audit['items']) and all(x['sha256_matches'] for x in audit['sources'])
    (ROOT / '来源与分类核验.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['pass', 'total_papers', 'total_pages', 'bc_pages']}, ensure_ascii=False))
    if not audit['pass']:
        raise RuntimeError('Source/category verification failed; inspect saved audit.')


if __name__ == '__main__':
    main()
