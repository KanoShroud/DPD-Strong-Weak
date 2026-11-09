"""Read back Zotero classification and XLSX export without changing either."""
import json
import argparse
import hashlib
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTIONS = {"A": "FC7V9KDA", "B": "47IRAJ5X", "C": "DEMZBKZG", "D": "LRSYGFUW"}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def workbook_rows(path):
    result = {}
    with zipfile.ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        for i in range(1, 5):
            doc = ET.fromstring(book.read(f"xl/worksheets/sheet{i}.xml"))
            sheet_count = 0
            for row in doc.findall(".//s:sheetData/s:row", NS):
                if int(row.get("r")) < 6:
                    continue
                cells = {}
                for cell in row.findall("s:c", NS):
                    val = cell.find("s:v", NS)
                    value = val.text if val is not None else ""
                    if cell.get("t") == "s":
                        value = strings[int(value)]
                    elif cell.get("t") == "inlineStr":
                        value = "".join(cell.find("s:is", NS).itertext())
                    col = "".join(c for c in cell.get("r") if c.isalpha())
                    cells[col] = value
                if cells.get("A"):
                    assert cells["A"] not in result
                    result[cells["A"]] = [cells.get(chr(65 + j), "") for j in range(13)]
                    sheet_count += 1
            table = ET.fromstring(book.read(f"xl/tables/table{i}.xml"))
            assert table.get("ref") == f"A5:M{5 + sheet_count}"
            assert table.find("s:autoFilter", NS).get("ref") == table.get("ref")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-workbook", type=Path)
    args = parser.parse_args()
    matrix = json.loads((HERE.parent / "近邻重审_20261105/文献矩阵.json").read_text(encoding="utf-8"))
    reviews = [x for x in matrix if x.get("reading_date") == "2026-11-06"]
    sources = {x["id"]: x for x in json.loads((HERE / "sources.json").read_text(encoding="utf-8"))}
    checks = []
    for row in reviews:
        key = row["zotero_keys"][0]
        req = urllib.request.Request(f"http://127.0.0.1:23119/api/users/0/items/{key}", headers={"Zotero-API-Version": "3"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            actual = json.load(resp)["data"]
        before = sources[row["id"]]["matches"][0]["data"]
        expected = COLLECTIONS[row["grade"]]
        assert set(actual["collections"]) & set(COLLECTIONS.values()) == {expected}, row["id"]
        assert set(before["collections"]) - set(COLLECTIONS.values()) <= set(actual["collections"]), row["id"]
        assert actual.get("tags", []) == before.get("tags", []), row["id"]
        checks.append({"id": row["id"], "item_key": key, "grade": row["grade"], "collection_verified": True, "other_collections_preserved": True, "tags_unchanged": True})
    with zipfile.ZipFile(HERE / "近邻文献精读矩阵.xlsx") as book:
        assert book.testzip() is None
        wb = ET.fromstring(book.read("xl/workbook.xml"))
        sheets = wb.findall("s:sheets/s:sheet", NS)
        assert len(sheets) == 4
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(book.read("xl/sharedStrings.xml"))]
        found = []
        panes = []
        for i in range(1, 5):
            doc = ET.fromstring(book.read(f"xl/worksheets/sheet{i}.xml"))
            for cell in doc.findall(".//s:sheetData/s:row/s:c", NS):
                address = cell.get("r")
                if not address.startswith("A") or int(address[1:]) < 6:
                    continue
                val = cell.find("s:v", NS)
                if cell.get("t") == "s":
                    found.append(strings[int(val.text)])
                elif cell.get("t") == "inlineStr":
                    found.append("".join(cell.find("s:is", NS).itertext()))
                elif val is not None:
                    found.append(val.text)
            pane = doc.find("s:sheetViews/s:sheetView/s:pane", NS)
            panes.append(pane.attrib if pane is not None else None)
        assert Counter(found) == Counter(r["id"] for r in reviews), (len(found), found)
        assert all(p and p.get("xSplit") == "2" and p.get("ySplit") == "5" for p in panes), panes
    actual_rows = workbook_rows(HERE / "近邻文献精读矩阵.xlsx")
    for entry in reviews:
        r = entry["close_reading"]
        expected = [entry["id"], f"{entry['title']}\n{entry.get('authors', '')}\n{entry['year']} · {entry.get('venue', '')}", r["scene"], r["observations_priors"], r["method"], r["theory"], r["overlap"], r["insight"], f"{r['baseline_role']}\n限制：{r['limits']}", r["experiments"], r["grade_reason"], r["read_scope"], f"{r['evidence']}\n{('https://doi.org/' + entry['doi']) if entry.get('doi') else entry['url']}\nZotero：{', '.join(entry['zotero_keys'])}"]
        assert actual_rows[entry["id"]] == expected, entry["id"]
    preserved = None
    if args.before_workbook:
        old_rows = workbook_rows(args.before_workbook)
        assert all(actual_rows[key] == value for key, value in old_rows.items())
        preserved = len(old_rows)
    identities = []
    for source in json.loads((HERE / "sources_batch23.json").read_text(encoding="utf-8")):
        for pdf in source["pdfs"]:
            digest = hashlib.sha256(Path(pdf["path"]).read_bytes()).hexdigest()
            assert digest == pdf["sha256"], source["id"]
            identities.append({"id": source["id"], "attachment": pdf["attachment_key"], "sha256": digest})
    result = {"papers": len(reviews), "sheets": [s.get("name") for s in sheets], "panes": panes, "all_13_columns_match_catalog": True, "previous_rows_preserved": preserved, "new_pdf_identities": identities, "zotero": checks, "status": "PASS"}
    (HERE / "交付核验.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(reviews)} unique paper rows, 4 worksheets, freeze panes, Zotero categories and unchanged tags")


if __name__ == "__main__":
    main()
