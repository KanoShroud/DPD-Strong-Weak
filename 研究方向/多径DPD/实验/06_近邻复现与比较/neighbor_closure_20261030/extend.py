"""Rate-limited retry and targeted local full-text audit cache."""
import json
import time
from pathlib import Path

import fitz

from collect import OUT, fetch

KEYS = ['N9VGD3QG', 'YWUTRNXK', '9IVDLEHJ', '4KD3RU7N', '2ENUNLHX',
        '5KMCQ3Z7', 'YSMGTJV8', 'IF3K344E', 'JXFE2X2D', 'PIX9ZHCJ',
        'LH3D9LYE', '9MUIS57E']
EXTRA = ['multipath direct localization Signal Processing',
         'multipath direct position determination Digital Signal Processing',
         'Parameter estimation superimposed signals EM Feder Weinstein',
         'Space alternating generalized expectation maximization Fessler Hero',
         'blind localization multiple sources unknown channels variable projection']


def main():
    prior = json.loads((OUT / 'search_raw.json').read_text(encoding='utf-8'))
    result = {}
    for i, q in enumerate(prior['queries']):
        if prior['results']['crossref_' + str(i)].get('status') != 200:
            time.sleep(1.2)
            result['retry_' + str(i)] = fetch('https://api.crossref.org/works',
                {'query.bibliographic': q, 'rows': 25})
            print('retry', i, result['retry_' + str(i)].get('status'), flush=True)
    for i, q in enumerate(EXTRA):
        time.sleep(1.2)
        result['extra_' + str(i)] = fetch('https://api.crossref.org/works',
            {'query.bibliographic': q, 'rows': 30})
        print('extra', i, result['extra_' + str(i)].get('status'), flush=True)
    (OUT / 'search_extended.json').write_text(json.dumps({'queries': EXTRA,
        'results': result}, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    for key in KEYS:
        r = fetch(f'http://127.0.0.1:23119/api/users/0/items/{key}/children')
        children = r.get('data', [])
        for x in children:
            d = x['data']
            if d.get('contentType') != 'application/pdf':
                continue
            path = Path(r'F:\Zotero\storage') / d['key'] / d.get('filename', '')
            if not path.is_file():
                continue
            with fitz.open(path) as pdf:
                (OUT / (key + '.txt')).write_text('\n\n'.join(
                    f'PDF PAGE {i+1}\n' + p.get_text() for i, p in enumerate(pdf)), encoding='utf-8')
                records.append({'key': key, 'attachment': d['key'], 'path': str(path),
                                'pages': len(pdf)})
            break
    (OUT / 'local_fulltexts.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', len(records), 'cached fulltexts; extraction does not imply all read')


if __name__ == '__main__':
    main()
