"""Resolve final citation-chain branches, without expanding the research scenario."""
import json
import time

from collect import OUT, fetch

dois = ['10.1109/TSIPN.2025.3600826', '10.1109/ISCTIS63324.2024.10699283',
        '10.1109/TIT.2025.3566885', '10.1016/j.dsp.2026.106007',
        '10.1016/j.dsp.2025.105353', '10.1049/rsn2.12408']
res = {}
for doi in dois:
    res[doi] = fetch('https://api.crossref.org/works/' + doi)
    time.sleep(.6)
res['stoica_title'] = fetch('https://api.crossref.org/works',
    {'query.title': 'On the identifiability of multipath parameters', 'rows': 3})
res['mimo_title'] = fetch('https://api.crossref.org/works',
    {'query.title': 'Direct Location Estimation for MIMO Systems in Multipath Environments', 'rows': 3})
(OUT / 'last_branch.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
for key, value in res.items():
    d = value.get('data', {}).get('message', {})
    print(key, value.get('status'))
    for item in d.get('items', [d]):
        print(item.get('DOI'), item.get('title'))
