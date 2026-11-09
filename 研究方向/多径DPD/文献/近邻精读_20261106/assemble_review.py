"""Merge explicitly scoped close readings; never infer reading from PDF presence."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "近邻重审_20261105"
GRADES = {"A": "一级｜核心场景与最邻近定位", "B": "二级｜源径关联与盲分离思想", "C": "三级｜不同信息条件的定位", "D": "四级｜选模求解与理论基础"}


def main():
    sources = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    source_by_id = {s["id"]: s for s in sources}
    reviews = []
    for name in ("reviews_core.json", "reviews_neighbors.json", "reviews_theory.json", "reviews_main.json", "reviews_batch23.json", "reviews_blind_batch23.json", "reviews_theory_batch23.json"):
        reviews.extend(json.loads((HERE / name).read_text(encoding="utf-8")))
    assert len(reviews) == len({r["id"] for r in reviews}) == 69
    assert {r["id"] for r in reviews} == set(source_by_id)
    required = ("grade", "scene", "observations_priors", "method", "theory", "experiments", "overlap", "insight", "baseline_role", "limits", "evidence", "read_scope")
    assert all(all(r.get(k) for k in required) for r in reviews)
    matrix = json.loads((OLD / "文献矩阵.json").read_text(encoding="utf-8"))
    if not any(r["id"] == "N69" for r in matrix):
        src = source_by_id["N69"]
        data = src["matches"][0]["data"]
        matrix.append({"id": "N69", "title": src["title"], "doi": data.get("DOI", ""), "url": data.get("url", ""), "year": "2026", "venue": data.get("publicationTitle", ""), "grade": "D", "authors": "; ".join(c.get("firstName", "") + " " + c.get("lastName", "") for c in data.get("creators", [])), "priority": "优先", "provenance": "用户2026-11-06新增，IEEE已接收作者版"})
    by_id = {r["id"]: r for r in reviews}
    for entry in matrix:
        if entry["id"] not in by_id:
            continue
        review = by_id[entry["id"]]
        src = source_by_id[entry["id"]]
        if not entry.get("authors"):
            entry["authors"] = "; ".join((c.get("firstName", "") + " " + c.get("lastName", c.get("name", ""))).strip() for c in src["matches"][0]["data"].get("creators", []) if c.get("creatorType") == "author")
        entry.setdefault("grade_before_close_reading", src["original_grade"])
        entry["grade"] = review["grade"]
        entry["grade_name"] = GRADES[review["grade"]]
        entry["close_reading"] = review
        entry["reading_date"] = "2026-11-06"
        entry["reading_batch"] = src["scope"]
        entry["information_conditions"] = review["observations_priors"]
        entry["idea"] = review["insight"]
        entry["evidence_scope"] = review["read_scope"]
        entry["action"] = "本轮正文精读完成；附录/勘误等边界见精读记录"
        entry["local_pdf"] = True
        entry["library_status"] = "在库且已读取本轮附件"
        entry["zotero_keys"] = [x["key"] for x in src["matches"]]
        entry["pdfs"] = [{"key": p["attachment_key"], "path": p["path"], "pages": p["pages"], "exists": True} for p in src["pdfs"]]
    (OLD / "文献矩阵.json").write_text(json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changes = [{"id": e["id"], "old": e["grade_before_close_reading"], "new": e["grade"], "reason": e["close_reading"]["grade_reason"]} for e in matrix if e["id"] in by_id and e["grade_before_close_reading"] != e["grade"]]
    summary = {"date": "2026-11-06", "read_count": len(reviews), "catalog_count": len(matrix), "read_grades": dict(Counter(r["grade"] for r in reviews)), "catalog_grades": dict(Counter(r["grade"] for r in matrix)), "grade_changes": changes, "excluded_this_round": [e["id"] for e in matrix if e["id"] not in by_id]}
    (HERE / "reading_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 当前近邻分级文献矩阵", "", "2026-11-06更新：目录69篇均已完成正文精读。此前48篇，续读第二批12篇、第三批9篇；长证明的选读范围、外置补充缺口及原文疑点逐篇注明，不等于算法复现或全部证明已独立复证。", "", "**批次是补全文的顺序，级别是与当前研究的关系；二者不一一对应。** 四级中的关键理论论文仍可优先阅读。", "", "详细场景、方法、重合性、启示、证据页码见[Excel精读矩阵](../近邻精读_20261106/近邻文献精读矩阵.xlsx)和[综合结论](../近邻精读_20261106/精读结论与下一步.md)。机器可读事实源为本目录文献矩阵.json。", ""]
    for grade, label in GRADES.items():
        lines += [f"## {label}", "", "| ID | 文献 | 本轮阅读范围 | 分级依据／信息边界 |", "|---|---|---|---|"]
        for e in matrix:
            if e["grade"] != grade:
                continue
            review = e.get("close_reading", {})
            status = review.get("read_scope", "本轮未读；保留此前书目登记")
            why = review.get("grade_reason", e.get("information_conditions", "待全文核验"))
            title = e["title"].replace("|", "/")
            url = e.get("url") or ("https://doi.org/" + e["doi"] if e.get("doi") else "")
            lines.append(f"| {e['id']} | [{title}]({url}) | {status.replace('|', '/')} | {why.replace('|', '/')} |")
        lines.append("")
    (OLD / "分级文献矩阵.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
