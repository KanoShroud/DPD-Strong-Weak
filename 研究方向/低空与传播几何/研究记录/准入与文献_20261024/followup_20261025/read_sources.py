"""Read newly supplied PDFs without changing Zotero or localization code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import fitz

ROOT = Path(__file__).resolve().parent
PREFLIGHT = Path('C:/Users/Administrator/.codex/skills/academic-research-suite/ars/scripts/pdf_read_preflight.py')
SOURCES = [('tmc2024', 'WJ88HLYD', 'KMGTEFF5'), ('twc2024', '5LSTRQ7R', 'CJ27SSFB')]


def main():
    records = []
    for slug, item, attachment in SOURCES:
        files = list((Path('F:/Zotero/storage') / attachment).glob('*.pdf'))
        if len(files) != 1:
            raise ValueError(f'Ambiguous attachment {attachment}')
        source = files[0].resolve(strict=True)
        folder = ROOT / slug
        folder.mkdir(parents=True, exist_ok=True)
        checked = subprocess.run([sys.executable, '-X', 'utf8', str(PREFLIGHT), str(source)],
                                 capture_output=True, text=True, encoding='utf-8', errors='replace',
                                 check=True, env={**os.environ, 'PYTHONUTF8': '1',
                                                  'PYTHONIOENCODING': 'utf-8:replace'})
        (folder / 'integrity.json').write_text(checked.stdout, encoding='utf-8')
        with fitz.open(source) as doc:
            pages = []
            for i, page in enumerate(doc, 1):
                content = page.get_text(sort=True)
                (folder / f'page_{i}.txt').write_text(content, encoding='utf-8')
                page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(folder / f'page_{i}.png')
                pages.append({'page': i, 'characters': len(content)})
            records.append({'slug': slug, 'item_key': item, 'attachment_key': attachment,
                            'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                            'total_pages': len(doc), 'pages': pages,
                            'structural_verdict': json.loads(checked.stdout)['verdict']})
    (ROOT / 'sources.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
