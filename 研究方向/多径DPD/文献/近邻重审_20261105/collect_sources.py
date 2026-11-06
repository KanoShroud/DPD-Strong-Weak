"""Read-only Zotero snapshot and page extraction; never alters the library."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import fitz

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sources'
OUT.mkdir(exist_ok=True)


def get(route):
    with urlopen('http://127.0.0.1:23119/api/users/0/' + route, timeout=30) as r:
        return json.load(r)


def main():
    items = []
    for offset in range(0, 20000, 100):
        part = get(f'items?limit=100&start={offset}')
        items.extend(part)
        if len(part) < 100:
            break
    else:
        raise RuntimeError('Unexpected library pagination size')
    (OUT / 'zotero_items.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
    collections = get('collections?limit=100')
    (OUT / 'collections_before.json').write_text(json.dumps(collections, ensure_ascii=False, indent=2), encoding='utf-8')
    candidates = [x['data'] for x in items if x['data'].get('itemType') not in ('attachment', 'note', 'annotation')]
    print(f'Zotero records={len(items)}, bibliographic parents={len(candidates)}')
    pdf = Path('F:/Zotero/storage/USPFK577/A direct localization method in the presence of cochannel interference and multipath fading-Xia 等-2021.pdf')
    doc = fitz.open(pdf)
    manifest = {'pdf': str(pdf), 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'pages': len(doc), 'page_files': []}
    for i, page in enumerate(doc):
        stem = f'xia_p{2855+i}'
        (OUT / (stem + '.txt')).write_text(page.get_text(sort=True), encoding='utf-8')
        page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7)).save(OUT / (stem + '.png'))
        manifest['page_files'].append(stem)
    (OUT / 'xia_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
