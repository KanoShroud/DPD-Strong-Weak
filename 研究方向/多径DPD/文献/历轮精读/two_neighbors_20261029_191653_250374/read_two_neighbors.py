"""Capture two newly supplied Zotero PDFs without modifying the library."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import urllib.request
import urllib.parse
import fitz


def main():
    out = Path(__file__).resolve().parent / ('two_neighbors_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    api = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = 'http://127.0.0.1:23119/api/users/0/items/'
    records = []
    for key in ['LYBR3SMA', 'K3U45IN8']:
        item = json.load(api.open(base + key, timeout=20))
        children = json.load(api.open(base + key + '/children', timeout=20))
        child = next(c for c in children if c['data'].get('contentType') == 'application/pdf')
        url = api.open(base + child['key'] + '/file/view/url', timeout=20).read().decode('utf-8')
        try:
            url = json.loads(url)
        except json.JSONDecodeError:
            pass
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = fitz.open(path)
        (out / (key + '.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n' + p.get_text(sort=True) for i, p in enumerate(doc)), encoding='utf-8')
        records.append(dict(key=key, metadata=item['data'], attachment=child['key'], path=str(path), sha256=digest, pages=len(doc)))
    (out / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(out)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
