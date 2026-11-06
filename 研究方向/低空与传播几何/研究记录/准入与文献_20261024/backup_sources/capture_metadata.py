"""Capture public Crossref identity metadata; no Zotero writes."""
from pathlib import Path
import json
import urllib.request
import urllib.parse
import ssl
import certifi

DOIS = ['10.1109/ACCESS.2019.2907136', '10.1109/MILCOM58377.2023.10356379', '10.1109/TVT.2020.3025386']
root = Path(__file__).resolve().parent
results = []
for doi in DOIS:
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi, safe='')
    with urllib.request.urlopen(url, timeout=30, context=ssl.create_default_context(cafile=certifi.where())) as response:
        data = json.load(response)
    results.append({'request': url, 'response': data})
(root / 'crossref_identity.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('Saved', len(results), 'Crossref records; this is metadata, not full text.')
