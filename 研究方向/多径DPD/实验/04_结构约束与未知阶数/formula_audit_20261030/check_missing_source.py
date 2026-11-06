"""One public publisher PDF attempt; preserve failure rather than claiming full text."""
from pathlib import Path
import json
import requests
import fitz

OUT = Path(__file__).resolve().parent
metadata = json.loads((OUT/'Friedlander2014_crossref.json').read_text(encoding='utf-8'))['message']
url = metadata['link'][0]['URL']
record = dict(url=url, source='Crossref publisher-provided link', fulltext_obtained=False)
try:
    response = requests.get(url, timeout=20)
    record.update(status=response.status_code, content_type=response.headers.get('Content-Type'),
                  bytes=len(response.content), final_url=response.url)
    if response.ok and response.content.startswith(b'%PDF'):
        doc = fitz.open(stream=response.content, filetype='pdf')
        text = '\n'.join(p.get_text() for p in doc)
        if 'geolocation' in text.lower() and 'friedlander' in text.lower():
            (OUT/'Friedlander2014_full.pdf').write_bytes(response.content)
            (OUT/'Friedlander2014_full.txt').write_text(text, encoding='utf-8')
            record['fulltext_obtained'] = True
except Exception as exc:
    record['error'] = str(exc)
(OUT/'missing_source_attempt.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
program = fitz.open(OUT/'Friedlander2014_program.pdf')
(OUT/'Friedlander2014_program_p68.txt').write_text(program[68].get_text(), encoding='utf-8')
print(record)
