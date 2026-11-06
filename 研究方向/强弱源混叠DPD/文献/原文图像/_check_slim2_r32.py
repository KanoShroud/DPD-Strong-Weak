# -*- coding: utf-8 -*-
"""临时校验：统计 _frag_slim2/R32-R37 的字数、字段上限、关系句数与风险符号。"""
import re
import pathlib

FIELDS = ['编号', '标题', '作者年份', '一句话概括', '研究对象与场景', '信号与观测模型',
          '是否已知源数', '核心方法', '对混合IQ的适用性', '实验条件', '主要结果',
          '可复现性', '论文结论与支持边界', '主要贡献', '主要局限', '总体评价',
          '与本研究的关系', '证据定位']
LIM = {'作者年份': 30, '一句话概括': 90, '研究对象与场景': 100, '信号与观测模型': 100,
       '是否已知源数': 40, '核心方法': 200, '对混合IQ的适用性': 90, '实验条件': 130,
       '主要结果': 160, '可复现性': 60, '论文结论与支持边界': 130, '主要贡献': 140,
       '主要局限': 140, '总体评价': 130, '与本研究的关系': 250, '证据定位': 80}
BAD = ['$', '^', 'Φ', 'φ', 'λ', 'Σ', '√', 'arg min', 'argmax', 'arg max', chr(92), '_{',
       'α', 'τ', 'η', 'θ', 'σ', 'μ', '≈', '∈', '≥', '≤', 'diag', 'log(', 'det(',
       'QR', 'vec(', '‖']


def nws(s):
    return re.sub(r'\s', '', s)


base = pathlib.Path(r'F:\SourceCount_DPD_E2E\outputs_e2e\_pdfrender\_frag_slim2')
for i in range(26, 38):
    p = base / ('R%d.md' % i)
    t = p.read_text(encoding='utf-8')
    lines = t.split('\n')
    got = [ln[2:].split(':')[0] for ln in lines if ln.startswith('@@')]
    tot = len(nws(t))
    raw = len(t)
    msgs = []
    if not (1800 <= tot <= 2500):
        msgs.append('总字数%d越界' % tot)
    if got != FIELDS:
        msgs.append('字段异常')
    for ln in lines:
        if '@@' in ln and not ln.startswith('@@'):
            msgs.append('行内@@')
        if not ln.startswith('@@'):
            continue
        name = ln[2:].split(':')[0]
        if ':' not in ln:
            msgs.append('%s缺冒号' % name)
            continue
        val = ln.split(':', 1)[1]
        n = len(nws(val))
        if name in LIM and n > LIM[name]:
            msgs.append('%s=%d>%d' % (name, n, LIM[name]))
        if name == '与本研究的关系':
            s4 = [x for x in re.split('。', val) if x.strip()]
            if len(s4) != 4:
                msgs.append('关系句数=%d' % len(s4))
            for k, s in enumerate(s4, 1):
                if len(nws(s)) > 60:
                    msgs.append('关系第%d句=%d>60' % (k, len(nws(s))))
    hits = [b for b in BAD if b in t]
    if hits:
        msgs.append('风险符号:' + ','.join(hits))
    print('R%d: 非空白=%d 原始len=%d 字段数=%d %s' % (i, tot, raw, len(got), (' | '.join(msgs) if msgs else 'PASS')))
