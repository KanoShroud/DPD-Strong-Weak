"""Merge audited bibliography records with a read-only Zotero snapshot."""
import html
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlparse

import fitz

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OLD = PROJECT / 'outputs_e2e/research_admission_20261027/localization_identifiability/neighbor_closure_20261030/近邻文献矩阵.md'
GRADES = {'A': '一级｜核心场景与最邻近定位', 'B': '二级｜源径关联与盲分离思想',
          'C': '三级｜不同信息条件的定位', 'D': '四级｜选模求解与理论基础'}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', html.unescape(s).lower())


def main():
    rows = []
    for line in OLD.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'\| (N\d+) \[(.*?)\]\((.*?)\) \| (.*?) \| (.*?) \| (.*?) \|', line)
        if not match:
            continue
        ident, title, url, meta, scope, info = match.groups()
        n = int(ident[1:])
        grade = 'A' if n in (1, 2, 3, 4, 6, 13, 14, 15, 16) else ('B' if n in (5, 8, 9, 10, 11, 12, 21, 22, 26, 31, 39, 41, 42, 43, 44) else ('D' if n in (36, 37, 38, 40, 45) else 'C'))
        doi = url.split('doi.org/', 1)[1] if 'doi.org/' in url else ''
        year = re.search(r'20\d{2}|19\d{2}', meta)
        key = re.search(r'；([A-Z0-9]{8})(?:$|；)', meta)
        rows.append(dict(id=ident, title=title, doi=doi, url=url, year=year.group() if year else '',
                         venue=meta.split('；')[0], grade=grade, information_conditions=info,
                         evidence_scope='历史2026-10-30记录：' + scope + '；本轮未逐篇重读',
                         priority='优先' if grade == 'A' or n in (8, 9, 26, 37, 38, 40, 41) else '次优先',
                         legacy_key=key.group(1) if key else '', provenance=str(OLD), idea=info))
    for filename in ['additions.json', 'search_expansion/新增近邻候选.json']:
        for r in json.loads((ROOT / filename).read_text(encoding='utf-8-sig')):
            existing = next((x for x in rows if r.get('doi') and x['doi'].lower() == r['doi'].lower()), None)
            if existing:
                existing.update(r)
            else:
                r['id'] = f'N{len(rows)+1:02d}'
                rows.append(r)
            r.setdefault('provenance', filename)
    for r in rows:
        r.setdefault('url', 'https://doi.org/' + r['doi'])
        r.setdefault('grade', 'B')
        title = r['title'].lower()
        if any(w in title for w in ('model selection for sinusoids', 'adaptive model order', 'sage', 'incremental automatic', 'separable optimization', 'super-resolution estimation of uwb')):
            r['grade'] = 'D'
        if any(w in title for w in ('passive radar', 'passive mimo radar', 'dual uavs', 'binary structured', 'cayley')):
            r['grade'] = 'C'
        if r['doi'].lower() == '10.1109/twc.2025.3554614':
            r['information_conditions'] = '本轮Zotero§III确认单源、每anchor为ULA、多子载波CSI；§VIII明确多源属未来扩展。不能直接作全部单天线未知双源基线。'
            r['evidence_scope'] = '本轮Zotero§III模型、§VIII边界及参考文献定向核读；未逐页全篇精读'
            r['priority'] = '优先'
        if r['doi'].lower() == '10.1109/jsyst.2020.3005918':
            r['evidence_scope'] = '本轮4页正文、公式、图表与引用全部重读并逐页视觉核验；见Xia原文逐项复核.md'
            r['priority'] = '优先'
        if r['id'] == 'N10':
            r['url'] = 'https://www.iss.uni-stuttgart.de/forschung/publikationen/kreissig_itg10.pdf'
            r['fulltext_url'] = r['url']
            r['venue'] = 'ITG-Fachtagung Sprachkommunikation, Paper 38'
            r['evidence_scope'] = '本轮作者机构PDF首页/摘要确认出处；库内原Conference on Natural Language Processing元数据错误，本轮未改元数据'
        if r['id'] == 'N37':
            r['fulltext_url'] = 'https://www.eng.tau.ac.il/~meir/articles/3%20Parameter%20Estimation%20of%20Superimposed%20Signals%20Using%20the%20EM.pdf'
        if r['id'] == 'N38':
            r['fulltext_url'] = 'https://web.eecs.umich.edu/~hero/Preprints/Space-Alternating.pdf'
            r['evidence_scope'] += '；本轮查到作者原文及勘误页https://web.eecs.umich.edu/~hero/det_est.html，后续证明核验须同时读勘误'
    items = json.loads((ROOT / 'sources/zotero_items.json').read_text(encoding='utf-8'))
    parents = [x for x in items if x['data'].get('itemType') not in ('attachment', 'note', 'annotation')]
    for r in rows:
        matches = [x for x in parents if r.get('doi') and x['data'].get('DOI', '').lower() == r['doi'].lower()]
        if not matches:
            matches = [x for x in parents if norm(x['data'].get('title', '')) == norm(r['title'])]
        if not matches and r.get('legacy_key'):
            matches = [x for x in parents if x['key'] == r['legacy_key']]
        r['zotero_keys'] = [x['key'] for x in matches]
        r['pdfs'] = []
        for item in items:
            d = item['data']
            if d.get('parentItem') not in r['zotero_keys'] or d.get('contentType') != 'application/pdf':
                continue
            p = d.get('path', '')
            enclosure = item.get('links', {}).get('enclosure', {}).get('href', '')
            if not p and enclosure.startswith('file:'):
                p = unquote(urlparse(enclosure).path).lstrip('/')
            if p.startswith('storage:'):
                p = str(Path('F:/Zotero/storage') / item['key'] / p[8:])
            path = Path(p)
            exists = bool(p) and path.is_file()
            pages, error = None, None
            if exists:
                try:
                    with fitz.open(path) as doc:
                        pages = len(doc) if not doc.needs_pass else None
                except Exception as exc:
                    error = str(exc)
            r['pdfs'].append(dict(key=item['key'], path=p, exists=exists, pages=pages, error=error, bytes=path.stat().st_size if exists else None))
        r['local_pdf'] = any(p['pages'] for p in r['pdfs'])
        r['library_status'] = '在库且PDF存在' if r['local_pdf'] else ('在库但未找到可读PDF' if matches else '当前快照未匹配入库')
        r['action'] = '等待逐篇精读/复核' if r['local_pdf'] else ('请导入；已有公开作者稿链接' if r.get('fulltext_url') else '请下载导入正式全文；本轮未取得完整正文')
        if r['id'] == 'N02':
            r['action'] = '本轮全文逐项复核已完成；等待信道来源补齐后决定新锚定实验'
        r['grade_name'] = GRADES[r['grade']]
        if matches:
            d = matches[0]['data']
            r['authors'] = '; '.join((x.get('firstName', '') + ' ' + x.get('lastName', '')).strip() or x.get('name', '') for x in d.get('creators', []))
        else:
            r['authors'] = ''
    assert len({x['doi'].lower() for x in rows if x['doi']}) == sum(bool(x['doi']) for x in rows)
    (ROOT / '文献矩阵.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = dict(total=len(rows), grade=dict(Counter(r['grade'] for r in rows)),
                  in_library=sum(bool(r['zotero_keys']) for r in rows), local_pdf=sum(r['local_pdf'] for r in rows),
                  need_fulltext=sum(not r['local_pdf'] for r in rows), new_to_old_matrix=len(rows)-46)
    (ROOT / 'matrix_summary.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding='utf-8')
    write_markdown(rows, counts)
    print(json.dumps(counts, ensure_ascii=False))


def write_markdown(rows, counts):
    lines = ['# 当前双源盲多径DPD：分级文献矩阵', '', '更新：2026-11-05。分级表示与当前场景的距离，不是论文质量排名；阅读优先级独立，四级理论文献也可能必须优先读。信息条件差异不影响入库价值，但会改变比较角色。', '',
             f'共{counts["total"]}个候选书目条目；{counts["in_library"]}个已匹配Zotero，{counts["local_pdf"]}个有实际存在的库内PDF。其余为补库/全文请求，详见《待导入与全文缺口.md》。', '',
             '**阶段状态：候选矩阵，不是全篇精读完成版。** 历史阅读、当前定向核读、全文视觉核验分别注明；摘要不能支持未核实的算法前提。所有核心近邻补齐后再完成统一精读和最终综述。', '']
    for grade, label in GRADES.items():
        lines += ['## ' + label, '', '| ID | 文献与出版信息 | 库内/全文 | 信息条件与借鉴点 | 证据范围 |', '|---|---|---|---|---|']
        for r in rows:
            if r['grade'] != grade:
                continue
            title = f'[{r["title"]}]({r["url"]})<br>{r["venue"]}，{r["year"]}'
            status = r['library_status'] + '<br>' + ','.join(r['zotero_keys'])
            info = r['information_conditions'] + '<br>借鉴：' + r['idea']
            lines.append('| ' + ' | '.join([r['id'], title, status, info.replace('|', '/'), r['evidence_scope'].replace('|', '/')]) + ' |')
        lines += ['']
    (ROOT / '分级文献矩阵.md').write_text('\n'.join(lines), encoding='utf-8')
    missing = ['# 待导入与全文缺口', '', '仅列当前Zotero快照没有实际PDF的条目；“未获取”不等于不存在开放版。正式发表版本优先；作者稿链接作为合法补充。旧库存在但缺PDF的只补附件，不重复建条目。未匹配条目已按DOI/规范化题名查库，仍可能存在不同标题的预印本，导入前建议Zotero去重。', '',
               '本轮只确认来源，不将元数据或摘要当全文精读。以下所有条目都保留在矩阵；先补“第一批”核心/机制论文，再补完整二三批。', '']
    for label, predicate in [('第一批：核心或优先机制', lambda r: r['grade'] == 'A' or r.get('priority', '').startswith(('优先', 'P1'))),
                             ('第二批：重要方法与理论', lambda r: r['grade'] in ('B', 'D') and not (r['grade'] == 'A' or r.get('priority', '').startswith(('优先', 'P1')))),
                             ('第三批：不同信息条件与背景', lambda r: r['grade'] == 'C' and not r.get('priority', '').startswith(('优先', 'P1')))]:
        missing += ['## ' + label, '', '| ID | 正式版本／原始书目 | 开放版本线索 | 为什么保留 |', '|---|---|---|---|']
        for r in rows:
            if r['local_pdf'] or not predicate(r):
                continue
            extra = f'[作者稿/机构全文]({r["fulltext_url"]})' if r.get('fulltext_url') else '本轮未取得完整正文'
            official = 'https://doi.org/' + r['doi'] if r['doi'] else r['url']
            missing.append(f'| {r["id"]} | [{r["title"]}]({official}) — {r["year"]} | {extra} | {r["idea"]} |')
        missing += ['']
    (ROOT / '待导入与全文缺口.md').write_text('\n'.join(missing), encoding='utf-8')


if __name__ == '__main__':
    main()
