"""Read-only Zotero capture and page-addressable PDF extraction for A1--A9."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote, urlparse
import fitz
import requests

ROOT = Path(__file__).resolve().parent
API = 'http://127.0.0.1:23119/api/users/0'

def get(route, **params):
    r = requests.get(API + route, params=params, timeout=30)
    r.raise_for_status()
    return r

def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def main():
    items = []
    for start in range(0, 10000, 100):
        batch = get('/items', start=start, limit=100).json()
        items.extend(batch)
        if len(batch) < 100:
            break
    refs = json.loads((ROOT.parent / 'deep_review_20261025/reference_recommendations.json').read_text(encoding='utf-8'))
    manifest = []
    for ref in refs:
        if not ref['id'].startswith('A'):
            continue
        matches = [x for x in items if (ref.get('doi') and x['data'].get('DOI','').lower() == ref['doi'].lower()) or norm(x['data'].get('title','')) == norm(ref['title'])]
        entry = {'id': ref['id'], 'title': ref['title'], 'doi': ref.get('doi'), 'items': []}
        for item in matches:
            attachments = [x for x in items if x['data'].get('parentItem') == item['key'] and x['data'].get('contentType') == 'application/pdf']
            rec = {'key': item['key'], 'metadata': item['data'], 'attachments': []}
            for attachment in attachments:
                value = get('/items/' + attachment['key'] + '/file/view/url').text.strip()
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                path = Path(unquote(urlparse(value).path).lstrip('/'))
                doc = fitz.open(path)
                prefix = ref['id'] + '_' + attachment['key']
                pages = [p.get_text(sort=True) for p in doc]
                (ROOT / (prefix + '.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n{p}' for i,p in enumerate(pages)), encoding='utf-8')
                rec['attachments'].append({'key': attachment['key'], 'path': str(path), 'pages': len(pages), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'text': prefix + '.txt'})
            entry['items'].append(rec)
        manifest.append(entry)
        print(json.dumps({'id':entry['id'], 'matches':[(i['key'],[(a['key'], a['pages']) for a in i['attachments']]) for i in entry['items']]},ensure_ascii=False))
    (ROOT/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
