"""Read-only Zotero PDF extraction and selected-page rendering for source review."""
from pathlib import Path
import hashlib
import json
import fitz

OUT = Path(__file__).resolve().parent
SOURCES = {'R117': 'HJ6VGXLU', 'R118': 'HGK5UHTA'}

def main():
    manifest = {}
    for label, key in SOURCES.items():
        files = list((Path('F:/Zotero/storage')/key).glob('*.pdf'))
        if len(files) != 1:
            raise ValueError((label, files))
        path = files[0]
        with fitz.open(path) as doc:
            pages = [page.get_text(sort=True) for page in doc]
            target = OUT/label
            target.mkdir(exist_ok=True)
            for number, content in enumerate(pages, 1):
                (target/f'p{number:02}.txt').write_text(content, encoding='utf-8')
            manifest[label] = dict(path=str(path), attachment=key, pages=len(pages),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                text_chars=[len(t) for t in pages])
    (OUT/'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
