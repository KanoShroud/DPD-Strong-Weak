"""Capture a user-supplied PDF and add only approved topic memberships."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import fitz

ROOT = Path(__file__).resolve().parent
ITEM = 'LCNZ95Z6'
ATTACHMENT = 'G6RGC7BD'
TARGETS = {'4LXLXLLD': '00 低空实测与需求', '9ZAUQ8GQ': '04 可辨识性',
           'P3R7ZER6': '90 不同信息条件'}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=30) as response:
        return json.load(response)


def write(name, data):
    (ROOT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', action='store_true')
    args = parser.parse_args()
    files = list((Path('F:/Zotero/storage')/ATTACHMENT).glob('*.pdf'))
    assert len(files) == 1
    pdf = files[0]
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    doc = fitz.open(pdf)
    assert len(doc) == 14
    (ROOT/'全文提取.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n'+p.get_text(sort=True)
                                           for i, p in enumerate(doc)), encoding='utf-8')
    for number in [4, 5, 8, 13]:
        doc[number-1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(ROOT/f'page_{number:02}.png')
    item_url = f'http://127.0.0.1:23119/api/users/0/items/{ITEM}'
    before = request(item_url)
    write('item_before.json', before)
    write('source_identity.json', dict(item=ITEM, attachment=ATTACHMENT, pdf=str(pdf),
                                      sha256=digest, pages=14, version='arXiv:2610.30053v1',
                                      date='2026-10-28', intended_collections=TARGETS))
    if args.classify:
        receipts = []
        for key, name in TARGETS.items():
            collection = request(f'http://127.0.0.1:23119/api/users/0/collections/{key}')
            assert collection['data']['name'] == name
            if key in before['data']['collections']:
                receipts.append(dict(collection=key, already_present=True))
                continue
            result = request('http://127.0.0.1:23120/mcp', {
                'jsonrpc': '2.0', 'id': len(receipts)+1, 'method': 'tools/call',
                'params': {'name': 'add_items_to_collection',
                           'arguments': {'collectionKey': key, 'itemKeys': [ITEM]}}})
            receipts.append(dict(collection=key, result=result))
            write('classification_receipts.json', receipts)
            assert 'error' not in result and not result.get('result', {}).get('isError', False), result
        after = request(item_url)
        write('item_after.json', after)
        expected = set(before['data']['collections']) | set(TARGETS)
        assert set(after['data']['collections']) == expected
        unchanged = all(before['data'][k] == after['data'].get(k)
                        for k in before['data'] if k not in ['collections', 'version', 'dateModified'])
        assert unchanged, 'Unexpected metadata or tag change'
        write('classification_audit.json', dict(verified=True, item=ITEM, added=TARGETS,
                                               previous_preserved=before['data']['collections'],
                                               other_metadata_and_tags_unchanged=unchanged))
        print('分类完成并回读核对', list(TARGETS.values()))
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == digest
    print('PDF', len(doc), 'pages; SHA256', digest)


if __name__ == '__main__':
    main()
