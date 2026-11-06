"""Approved within-project direction layout. No algorithm/data changes."""
import argparse
import ast
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
WEAK = '研究方向/强弱源混叠DPD'
FINITE = '研究方向/有限记录DPD'
LOW = '研究方向/低空与传播几何'
MP = '研究方向/多径DPD'
MAINT = '项目维护'
RULES = {
 '统一模型代码/research/mp_dpd': MP+'/代码/mp_dpd',
 '统一模型代码/gates/g11': FINITE+'/代码/gates/g11',
 '统一模型代码/gates/research_discovery': FINITE+'/代码/gates/research_discovery',
 '统一模型代码': WEAK+'/代码',
 'DPD_MVDR': WEAK+'/代码/DPD_MVDR',
 '运行入口/多径DPD': MP+'/运行入口',
 '运行入口/E2E/G11': FINITE+'/运行入口/G11',
 '运行入口/E2E': WEAK+'/运行入口',
 '运行入口/__init__.py': MAINT+'/布局迁移_20261106/原始包标记/入口__init__.py',
 'paper_short_vp': WEAK+'/论文/VP短文',
 'paper': WEAK+'/论文/AP_NOMP双语稿',
 'outputs_e2e/unified/e2e_g11': FINITE+'/实验/G11',
 'outputs_e2e/unified': WEAK+'/实验',
 'outputs_e2e/research_discovery_20261022': FINITE+'/研究记录/方向筛选_20261022',
 'outputs_e2e/research_admission_20261024': LOW+'/研究记录/准入与文献_20261024',
 'outputs_e2e/research_admission_20261025/core_multipath': MP+'/实验/01_场景与模型/双源多径_20261025',
 'outputs_e2e/research_admission_20261025': LOW+'/实验/20261025',
 'outputs_e2e/research_admission_20261028': MP+'/文献/补充精读_20261028',
 'outputs_e2e/literature_reaudit_20261105/archive_plan': MAINT+'/首次目录盘点_20261105',
 'outputs_e2e/literature_reaudit_20261105': MP+'/文献/近邻重审_20261105',
 'outputs_e2e/mp_dpd/migration_20261106': MAINT+'/运行闭包迁移_20261106',
 'outputs_e2e/mp_dpd/history/00_早期公共脚本与协议': MP+'/研究记录/早期公式与协议',
 'outputs_e2e/mp_dpd/history/00_汇总报告': MP+'/研究记录/阶段汇总',
 'outputs_e2e/mp_dpd/history/01_可辨识性与局部求解': MP+'/实验/02_可辨识性与局部求解',
 'outputs_e2e/mp_dpd/history/02_盲初始化与源关联': MP+'/实验/03_盲初始化与源关联',
 'outputs_e2e/mp_dpd/history/03_结构约束与未知阶数': MP+'/实验/04_结构约束与未知阶数',
 'outputs_e2e/mp_dpd/history/04_位置风险与回放': MP+'/实验/05_位置风险与回放',
 'outputs_e2e/mp_dpd/history/05_近邻复现与比较': MP+'/实验/06_近邻复现与比较',
 'outputs_e2e/mp_dpd/history/06_提速与四站验证': MP+'/实验/07_提速与四站验证',
 'outputs_e2e/mp_dpd/history/07_总体架构与正则化': MP+'/实验/08_总体架构与正则化',
 'outputs_e2e/mp_dpd/runs': MP+'/实验/单条运行',
 'outputs_e2e/mp_dpd/checks': MP+'/实验/工程检查',
 'outputs_e2e/literature': WEAK+'/文献/精读与核验',
 'outputs_e2e/_pdfrender': WEAK+'/文献/原文图像',
 'outputs_e2e/document_review': WEAK+'/研究记录/文稿检查',
 'outputs_e2e/papers': WEAK+'/论文/前期稿件',
 'outputs_e2e/reports': WEAK+'/研究记录/专题报告',
 'outputs_e2e/verification': WEAK+'/实验/工程核验',
 'outputs_e2e/maintenance': MAINT+'/既有维护记录',
 'docs/mp_dpd/README.md': MP+'/研究记录/轮次导航.md',
 'docs': MAINT+'/路径与身份',
 'tools/mp_migration_probe.py': MP+'/验证/migration_probe.py',
 'tools/verify_mp_migration.py': MAINT+'/运行闭包迁移_20261106/verify.py',
 '项目拆分导航.md': MAINT+'/项目拆分导航.md',
 '参考文献编号对照.csv': WEAK+'/文献/参考文献编号对照.csv',
 '文献精读结果.csv': WEAK+'/文献/文献精读结果.csv',
 '文献精读结果_详版.csv': WEAK+'/文献/文献精读结果_详版.csv',
 '端到端多源定位_文献综述与方案建议.md': WEAK+'/文献/端到端多源定位_文献综述与方案建议.md',
 '统一端到端模型研究.md': WEAK+'/研究记录/前期物理定位研究.md',
 'Guide for authors - Signal Processing.pdf': MAINT+'/投稿规范/Guide for authors - Signal Processing.pdf',
}
MIXED = ('当前任务.md', '对话上下文.md', '实验结果记录.md', '研究总结.md', '修改记录.md')
EXEMPT = {'.git', '.idea', '.venv', '项目维护', '研究方向'}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.pending_layout')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def target(rel):
    for before, after in sorted(RULES.items(), key=lambda x: len(x[0]), reverse=True):
        if rel == before or rel.startswith(before+'/'):
            return after+rel[len(before):]
    if rel in MIXED:
        return '项目维护/布局迁移_20261106/原始混合文档/'+rel
    return None


def safe(path):
    p = path.resolve()
    if not p.is_relative_to(ROOT) or p == ROOT or p.relative_to(ROOT).parts[0] in ('.git', '.venv', '.idea'):
        raise ValueError(f'Unsafe path: {p}')
    return p


def plan():
    # Reading-only literature rounds belong to literature, not solver results.
    base = ROOT/'outputs_e2e/mp_dpd/history/05_近邻复现与比较'
    for p in base.iterdir():
        if p.is_dir() and any(x in p.name for x in ('literature', 'neighbor_review', 'two_neighbors')):
            RULES[p.relative_to(ROOT).as_posix()] = MP+'/文献/历轮精读/'+p.name
    rows, deleted, unknown = [], [], []
    for folder, dirs, files in os.walk(ROOT):
        path = Path(folder)
        if path == ROOT:
            dirs[:] = [d for d in dirs if d not in EXEMPT]
        for f in files:
            p = path/f
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink():
                raise RuntimeError(p)
            dest = target(rel)
            if not dest:
                if p.parent != ROOT and not any(v in p.parts for v in ('__pycache__', '.ruff_cache')):
                    unknown.append(rel)
                continue
            row = dict(old=rel, new=dest, bytes=p.stat().st_size)
            reason = None
            if any(v in p.parts for v in ('__pycache__', '.ruff_cache')) or p.suffix == '.pyc':
                reason = '可再生缓存'
            if rel.startswith(('paper/', 'paper_short_vp/')) and p.suffix in ('.aux', '.log', '.out', '.blg', '.fls', '.fdb_latexmk', '.gz'):
                reason = 'LaTeX编译中间文件；保留源码、PDF、参考文献与图表'
            if rel.startswith('运行入口/E2E/G11/') and p.suffix == '.py' and '历史批量入口已封存' in p.read_text(encoding='utf-8-sig'):
                reason = '重复停用提示；原入口源码已在研究记录保留'
            if reason:
                row['reason'] = reason
                deleted.append(row)
            else:
                if safe(ROOT/dest).exists():
                    raise RuntimeError(f'Destination exists: {dest}')
                rows.append(row)
    if unknown:
        print('UNCLASSIFIED', unknown[:40])
        raise RuntimeError('Unclassified files remain')
    assert len({r['new'].casefold() for r in rows}) == len(rows), 'Duplicate destinations'
    save(HOME/'映射.json', dict(status='planned', rules=RULES, files=len(rows), bytes=sum(r['bytes'] for r in rows),
        cleanup=deleted, modified_originals={}, split_sections=[]))
    with (HOME/'files.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print('PLAN', len(rows), 'files', round(sum(r['bytes'] for r in rows)/2**30,2), 'GiB', len(deleted), 'cleanup')


def move():
    info = json.loads((HOME/'映射.json').read_text(encoding='utf-8'))
    assert info['status'] == 'planned'
    rows = [json.loads(line) for line in (HOME/'files.jsonl').read_text(encoding='utf-8').splitlines()]
    for index, row in enumerate(rows):
        src, dst = safe(ROOT/row['old']), safe(ROOT/row['new'])
        if dst.exists() and not src.exists() and dst.stat().st_size == row['bytes']:
            continue
        if dst.exists() or src.stat().st_size != row['bytes']:
            raise RuntimeError(f'Source drift or collision: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Same-volume rename avoids making a second 23 GiB copy. Verify each
        # destination; original scientific bytes are never rewritten here.
        src.rename(dst)
        if dst.stat().st_size != row['bytes']:
            raise RuntimeError(f'Size changed: {dst}')
        if index % 20000 == 0:
            print('MOVED', index, flush=True)
    for row in info['cleanup']:
        src = safe(ROOT/row['old'])
        if src.exists():
            if src.stat().st_size != row['bytes']:
                raise RuntimeError(src)
            src.unlink()
    for first in ('统一模型代码', '运行入口', 'outputs_e2e', 'paper', 'paper_short_vp', 'DPD_MVDR', 'docs', 'tools'):
        base = ROOT/first
        if not base.exists():
            continue
        folders=[]
        for here, children, _ in os.walk(base):
            children[:]=[d for d in children if not (getattr((Path(here)/d).lstat(),'st_file_attributes',0)&0x400)]
            folders.append(Path(here))
        for p in sorted(folders, key=lambda x: len(x.parts), reverse=True):
            if p != base and not any(p.iterdir()):safe(p).rmdir()
        if not any(base.iterdir()):
            safe(base).rmdir()
    assert all((ROOT/r['new']).stat().st_size == r['bytes'] for r in rows)
    info['status'] = 'moved_verified'
    save(HOME/'映射.json', info)
    print('ALL MOVED AND SIZE VERIFIED', len(rows))


def sanitize():
    info=json.loads((HOME/'映射.json').read_text(encoding='utf-8'))
    rows=[json.loads(x) for x in (HOME/'files.jsonl').read_text(encoding='utf-8').splitlines()]
    kept=[];mounts={};excluded=0
    def outside(row):
        p=ROOT/row['old']
        if p.resolve().is_relative_to(ROOT):return False
        for parent in p.parents:
            if parent==ROOT:break
            if getattr(parent.lstat(),'st_file_attributes',0)&0x400:
                mounts[parent.relative_to(ROOT).as_posix()]=str(parent.resolve())
                break
        return True
    for r in rows:
        if outside(r):excluded+=1
        else:kept.append(r)
    info['cleanup']=[r for r in info['cleanup'] if not outside(r)]
    info['external_runtime_links']=mounts
    info['excluded_external_files']=excluded
    info['files']=len(kept);info['bytes']=sum(r['bytes'] for r in kept)
    with (HOME/'files.jsonl').open('w',encoding='utf-8') as stream:
        for row in kept:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    save(HOME/'映射.json',info)
    print('PROJECT FILES',len(kept),'GiB',round(info['bytes']/2**30,2),'EXTERNAL LINK ENTRIES EXCLUDED',excluded)
    print(json.dumps(mounts,ensure_ascii=False,indent=2))


def direction(block):
    head = block.split('\n', 1)[0]
    if re.search(r'G(?:8|9|10)(?:\D|$)|VP短文|长文审稿|投稿|2026-10-(?:0\d|1\d|20|21)', head):
        return WEAK
    if re.search(r'G11|有限记录|Slepian|Hann|2026-10-(?:22|23)|路线终审', head):
        return FINITE
    if re.search(r'墙面|反射面|有限立面|低空|无人机|2026-10-24', head):
        return LOW
    if re.search(r'2026-1[01]|20261[01]|多径|风险|Xia|源.*路径|正则|四站|盲|条件评分|后端|结构|位置', head):
        return MP
    # Undated early manuscript context follows the old AP/NOMP project.
    if re.search(r'G11|research_discovery|有限记录|Slepian|Hann', block):
        return FINITE
    return WEAK


def split():
    info = json.loads((HOME/'映射.json').read_text(encoding='utf-8'))
    for name in MIXED:
        src = HOME/'原始混合文档'/name
        text = src.read_text(encoding='utf-8-sig')
        if '\n## ' in text:
            blocks = re.split(r'(?m)(?=^## )', text)[1:]
        else:
            blocks = [b+'\n\n' for b in text.split('\n\n')[1:] if b.strip()]
        groups = {}
        for b in blocks:
            topic = direction(b)
            groups.setdefault(topic, []).append(b)
            info['split_sections'].append(dict(source=name, heading=b.split('\n',1)[0], topic=topic))
        for topic, parts in groups.items():
            dest = ROOT/topic/'研究记录'/('任务与决定沿革.md' if name == '当前任务.md' else name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError(dest)
            dest.write_text('# '+dest.stem+'\n\n按研究方向摘分；标题中的“当前”指当时状态，不代表新的运行授权。\n\n'+''.join(parts), encoding='utf-8')
    save(HOME/'映射.json', info)
    print('SPLIT', dict(Counter(r['topic'] for r in info['split_sections'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['plan','move','split','sanitize'])
    globals()[p.parse_args().phase]()
