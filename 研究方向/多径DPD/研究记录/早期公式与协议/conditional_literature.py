"""Read-only Zotero/public-source retrieval for the bounded follow-up review."""
from pathlib import Path
import hashlib
import json
import urllib.request
import urllib.parse
import fitz
import ssl
import certifi
import argparse

OUT = Path(__file__).parent/'conditional_20261029_112305_738322'/'literature'
LOCAL = ['2CI8A8RL', 'XWJMZUKE', 'JXFE2X2D', 'J2ZDTX8M', 'FH72DMFX', '4KD3RU7N', 'N9VGD3QG']
PUBLIC = {
    'TCBSS2024': ('10.1109/ICASSP48485.2024.10447269', 'https://thanhtbt.github.io/files/2024_ICASSP_TCBSS.pdf'),
    'Florian2012': ('10.1016/j.sigpro.2011.09.013', 'https://arxiv.org/pdf/1102.2706'),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--public-only', action='store_true')
    args = ap.parse_args()
    out = OUT/'public_retry' if args.public_only else OUT
    out.mkdir(exist_ok=False)
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def api(route):
        with local.open('http://127.0.0.1:23119/api/users/0/'+route, timeout=30) as response:
            return json.load(response)
    records = []
    for key in ([] if args.public_only else LOCAL):
        item = api('items/'+key); children = api('items/'+key+'/children')
        child = next(c for c in children if c['data'].get('contentType') == 'application/pdf')
        with local.open('http://127.0.0.1:23119/api/users/0/items/'+child['key']+'/file/view/url', timeout=30) as response:
            raw = response.read().decode()
        # Zotero endpoint returns plain file:// URL, not bibliographic content.
        if raw.startswith('"'): raw = json.loads(raw)
        path = Path(urllib.request.url2pathname(urllib.parse.urlsplit(raw).path))
        if not path.exists() and str(path).startswith('\\'):
            path = Path(str(path)[1:])
        assert path.exists(), (key, raw, path)
        records.append(dict(key=key, source='Zotero existing PDF', metadata=item['data'], path=str(path)))
    for key, (doi, url) in PUBLIC.items():
        record = dict(key=key, doi=doi, url=url)
        try:
            req = urllib.request.Request('https://api.crossref.org/works/'+doi, headers={'User-Agent': 'DPD-evidence-audit/1.0'})
            with urllib.request.urlopen(req, timeout=35, context=ssl.create_default_context(cafile=certifi.where())) as response:
                record['crossref'] = json.load(response)['message']
        except Exception as exc:
            record['crossref_error'] = repr(exc)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=45, context=ssl.create_default_context(cafile=certifi.where())) as response:
                data = response.read()
            assert data.startswith(b'%PDF'), 'not PDF'
            path = out/(key+'.pdf'); path.write_bytes(data)
            record['path'] = str(path)
        except Exception as exc:
            record['download_error'] = repr(exc)
        records.append(record)
    for row in records:
        if 'path' not in row: continue
        path = Path(row['path']); row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = fitz.open(path); row['pages'] = len(doc)
        pages = [p.get_text(sort=True) for p in doc]
        (out/(row['key']+'.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n{s}' for i, s in enumerate(pages)), encoding='utf-8')
        row['text_extract'] = row['key']+'.txt'
        print(row['key'], len(doc), 'pages', len(''.join(pages)), 'characters', flush=True)
    (out/'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__': main()
