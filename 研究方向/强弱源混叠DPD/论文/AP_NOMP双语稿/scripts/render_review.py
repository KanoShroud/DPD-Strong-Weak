"""Render every manuscript page and write geometry checks for visual review."""
from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, default=HERE/'paper_draft.pdf')
    parser.add_argument('--out', type=Path, default=HERE/'qa/current_review')
    args = parser.parse_args()
    pdf = args.pdf.resolve(strict=True)
    review = args.out.resolve()
    out = review / 'pages'
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    pages = []
    anomalies = []
    for number, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        target = out / f'page_{number:02d}.png'
        pix.save(target)
        text = page.get_text()
        if '\ufffd' in text or '??' in text:
            anomalies.append(dict(page=number, issue='Replacement or unresolved-reference text'))
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    if min(x0, y0) < 5 or x1 > page.rect.width-5 or y1 > page.rect.height-5:
                        anomalies.append(dict(page=number, issue='Text near/outside page edge', text=span['text']))
        pages.append(dict(page=number, text_characters=len(text), image=str(target.relative_to(HERE))))
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    for first in range(0, len(pages), 4):
        sheet = Image.new('RGB', (1200, 1760), '#d8d8d8')
        draw = ImageDraw.Draw(sheet)
        for j, p in enumerate(pages[first:first+4]):
            with Image.open(HERE / p['image']) as im:
                im.thumbnail((580, 830))
                x, y = (j % 2)*600+10, (j // 2)*880+35
                sheet.paste(im, (x, y))
                draw.text((x, y-28), f"Page {p['page']}", font=font, fill='black')
        sheet.save(review / f'contact_{first//4+1:02d}.png')
    report = dict(pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), pages=pages,
                  count=len(pages), text_geometry_anomalies=anomalies,
                  automated_passed=not anomalies,
                  visual_review='Human review must be recorded separately for this PDF hash; rendering alone is not visual certification.')
    (review/'render_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Rendered all {len(pages)} pages. Text-geometry anomalies: {len(anomalies)}.')
    if anomalies:
        raise AssertionError(anomalies)


if __name__ == '__main__':
    main()
