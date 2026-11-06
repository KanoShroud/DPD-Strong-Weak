"""Read-only Zotero PDF extraction and selected-page renders for admission audit."""
from pathlib import Path
import hashlib
import json
import fitz

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'oispuu2010': ('CUL7569J', [3, 4, 6, 8]),
    'you2025': ('E5999LAT', [2, 3]),
    'tirer2017': ('MQSLYJW4', [2, 5]),
    'bicer2026': ('3CJY59PS', [3]),
    'wang2025': ('8FEBBSJP', [3, 4]),
    'weiss2024': ('6RID5AXS', [2, 3]),
    'you2023': ('62NRERG3', [2, 3]),
    'ding2026mmse': ('852SAJ6P', [3]),
    'ding2026async': ('MU2TQ947', [3]),
}

def main():
    manifest = []
    for label, (key, pages) in SOURCES.items():
        matches = list((Path('F:/Zotero/storage') / key).glob('*.pdf'))
        if len(matches) != 1:
            raise ValueError((key, matches))
        path = matches[0]
        out = ROOT / label
        out.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(path)
        text = '\n'.join(f'\n=== PDF PAGE {i+1} ===\n{p.get_text()}' for i, p in enumerate(doc))
        (out / 'extracted.txt').write_text(text, encoding='utf-8')
        for page in pages:
            doc[page-1].get_pixmap(matrix=fitz.Matrix(1.35, 1.35)).save(out / f'page_{page}.png')
        manifest.append(dict(label=label, attachment=key, path=str(path), pages=len(doc),
                             rendered_pages=pages, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (ROOT / 'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
