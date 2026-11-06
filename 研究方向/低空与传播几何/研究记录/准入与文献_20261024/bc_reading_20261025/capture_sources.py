"""Capture approved bibliography and PDF text using read-only Zotero API."""
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import fitz
import requests

ROOT = Path(__file__).resolve().parent
API = 'http://127.0.0.1:23119/api/users/0'


def get(route, **params):
    response = requests.get(API + route, params=params, timeout=30)
    response.raise_for_status()
    return response


def norm(value):
    return re.sub(r'[^a-z0-9]', '', value.lower())


def main():
    items = []
    for start in range(0, 20000, 100):
        batch = get('/items', start=start, limit=100).json()
        items.extend(batch)
        if len(batch) < 100:
            break
    refs = json.loads((ROOT.parent / 'deep_review_20261025/reference_recommendations.json').read_text(encoding='utf-8'))
    manifest = []
    before = {}
    for ref in refs:
        matches = [i for i in items if i['data'].get('itemType') != 'attachment' and
                   ((ref.get('doi') and i['data'].get('DOI', '').lower() == ref['doi'].lower()) or
                    norm(i['data'].get('title', '')) == norm(ref['title']))]
        entry = {'id': ref['id'], 'title': ref['title'], 'doi': ref.get('doi'), 'items': []}
        for item in matches:
            before[item['key']] = item['data']
            record = {'key': item['key'], 'metadata': item['data'], 'attachments': []}
            for att in items:
                if att['data'].get('parentItem') != item['key'] or att['data'].get('contentType') != 'application/pdf':
                    continue
                raw = get('/items/' + att['key'] + '/file/view/url').text.strip()
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    pass
                path = Path(unquote(urlparse(raw).path).lstrip('/'))
                doc = fitz.open(path)
                filename = f"{ref['id']}_{att['key']}.txt"
                if ref['id'][0] in 'BC':
                    (ROOT / filename).write_text('\n'.join(f'\n=== PDF PAGE {p.number + 1} ===\n{p.get_text()}' for p in doc), encoding='utf-8')
                record['attachments'].append({'key': att['key'], 'path': str(path), 'pages': len(doc), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text': filename})
            entry['items'].append(record)
        manifest.append(entry)
    (ROOT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'zotero_items_before.json').write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'collections_before.json').write_text(get('/collections', limit=100).text, encoding='utf-8')
    for ref in manifest:
        if ref['id'][0] in 'BC':
            print(json.dumps({'id': ref['id'], 'title': ref['title'], 'items': [(r['key'], [(a['key'], a['pages'], a['path']) for a in r['attachments']]) for r in ref['items']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
