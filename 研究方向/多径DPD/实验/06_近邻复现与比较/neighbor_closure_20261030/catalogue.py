"""Build evidence-tiered bibliography; never writes to Zotero."""
import json
import re

from collect import OUT

# Identifier, family, evidence level, decision. The full-text reading scope is explicit.
ROWS = [
 ('KGA2WLPL','A 核心模型/框架','本轮全文5页＋公式页图核对','单个未知波形；未知共同反射体位置；复增益；EM。不能把多路径K误作独立源数。'),
 ('FRE2JXRP','A 核心模型/框架','复用此前全文及代码核验','同频未知多源、统计复多径、单天线站；当前最直接的场景近邻。'),
 ('FH72DMFX','A 核心模型/框架','复用上一轮§2/4公式审计','未知多源，已知转发器几何，非负实路径增益；有投影ML与初始化。'),
 ('69EEYHFF','A 核心模型/框架','复用上一轮§II/III全文核读','已知转发器几何的统一子空间拟合；不是自由复多径模型。'),
 ('CZPZD5NA','A 核心模型/框架','复用上一轮§II–IV核读','单源未知波形/复增益，水深决定路径时延；不能直接搬小矩阵公式。'),
 ('N9VGD3QG','A 核心模型/框架','本轮PDF第2页渲染核对；未全篇重读','多用户未知数据；已知成形脉冲和主要散射体位置；阵列时空处理。'),
 ('9IVDLEHJ','A 核心模型/框架','本轮PDF第3–4页§II/III核读','单UE，ULA，已知导频得到CSI；NLOS参数作辅助量，EM/AP；同思想非同信息。'),
 ('6YQ38NQH','B 几何关联/初始化','复用作者源码与共同输入配对','TDOA多峰关联；已有固定K LP可替代当前前端，不再主张前端独创。'),
 ('A9X2TH3D','B 几何关联/初始化','复用已有全文记录','多源混响TDOA歧义处理；几何关联思想近邻，不能因声学排除。'),
 ('W7YUTHCP','B 几何关联/初始化','库内元数据；未取得PDF','TDOA图一致性理论基础；不作为当前完整IQ方法性能参照。'),
 ('10.1109/lsp.2023.3328273','B 几何关联/初始化','作者机构摘要核验','发射传感器照射被动目标，输入为多组距离；不是未知辐射源盲TDOA。'),
 ('10.1109/tsipn.2025.3600826','B 几何关联/初始化','正式元数据＋摘要线索；全文待核','上项遮挡/未知目标数扩展；保留关联机制近邻，不借其摘要证明同场景。'),
 ('JXFE2X2D','C 不同信息条件的定位','本轮PDF第3页§II核读＋既有记录','公式为一个s(t)，K为接收分段；多TDOA不是多个独立源。'),
 ('XWJMZUKE','C 不同信息条件的定位','本轮出版社正文＋既有全文记录','单源跨谱/参考观测路线；不能把双源混合参考当作干净参考。'),
 ('CX82UIQA','C 不同信息条件的定位','本轮出版社方法节＋既有全文记录','单源跨谱平滑/子空间/Taylor补偿；不是我们的双源IQ联合拟合。'),
 ('YNYYLNVP','C 不同信息条件的定位','复用已有全文记录；本轮引文追踪','PARAFAC跨相关TDOA延续线，须保留其参考/单源前提。'),
 ('YWUTRNXK','C 不同信息条件的定位','本轮PDF第1–2页§II核读','同频但不同符号率/循环频率；当前两个同符号率BPSK不能默认可切片分开。'),
 ('IF3K344E','C 不同信息条件的定位','本轮PDF第2页§II核读','单目标已知发射脉冲，统计密集多径；非未知双源波形。'),
 ('5KMCQ3Z7','C 不同信息条件的定位','本轮模型/导频假设定向核读','移动多天线接收，已知OFDM导频及角度/Doppler信息。'),
 ('2ENUNLHX','C 不同信息条件的定位','本轮§II明确前提核读','已知波形、混合天线阵列、单发射机；不是直接公平盲基线。'),
 ('9MUIS57E','C 不同信息条件的定位','全文已提取，复用此前已知波形核验','分布式多径稀疏直接定位；波形已知。'),
 ('2CI8A8RL','C 不同信息条件的定位','复用§II全文核验','已知多源波形；库中为arXiv版本，未另确认正式同题版本。'),
 ('YSMGTJV8','C 不同信息条件的定位','本轮PDF第3页§II核读','多QAM源、每站ULA、频率平坦反射、四阶统计；不能去掉阵列仍称原法。'),
 ('2MIXHX3L','C 不同信息条件的定位','本轮出版社摘要/方法介绍；全文本轮未读','电磁矢量传感器及张量结构，额外观测维数不属于单标量天线。'),
 ('4KD3RU7N','C 不同信息条件的定位','本轮PDF第3页§II核读','多源单天线，但输入是RSS/频谱地图；不等于原IQ时延盲定位。'),
 ('J2ZDTX8M','C 不同信息条件的定位','复用已核读记录，本轮未重新精读','多径定位与关联SBL；其信息条件按旧全文记录使用，不仅按标题纳入。'),
 ('PIX9ZHCJ','C 不同信息条件的定位','本轮出版社模型摘要＋既有记录','多相干源和阵列；相干源DOA并非未知独立波形的逐链路延迟副本。'),
 ('LH3D9LYE','C 不同信息条件的定位','本轮PDF第1页模型核读','阵列多源谱/时空DPD，式(1)为LOS；不直接处理本项目未知多径。'),
 ('10.1016/j.dsp.2025.105290','C 不同信息条件的定位','本轮出版社模型节可见内容','多宽带源，ULA与滤波器组、DOA/TOA张量；全文尚未读。'),
 ('10.1016/j.dsp.2025.105353','C 不同信息条件的定位','本轮出版社方法/结论可见内容','JATD/CPD相干分布源，结论明确单CD源；不等于双源统计多径。'),
 ('10.1016/j.sigpro.2020.107907','C 不同信息条件的定位','作者稿PDF第9页式(6)及脚注5核读','移动多天线接收机自定位，符号已知；用估计多径构造投影的思路相关。'),
 ('10.1109/glocom.2008.ecp.646','C 不同信息条件的定位','作者PDF模型/ML节定向核读','收发MIMO、共同散射体、AOA/AOD与移动信息；非本场景直接对手。'),
 ('10.1016/j.sigpro.2026.110527','C 不同信息条件的定位','本轮出版社摘要','低空雷达阵列相干DOA、未知地形反射；不能因为标题无先验就认定同问题。'),
 ('10.1109/isctis63324.2024.10699283','C 不同信息条件的定位','元数据＋后续论文引用；全文待核','分布式阵列多径矩阵重构；只列补充，未作公式不重合结论。'),
 ('10.1016/j.sigpro.2014.02.010','C 不同信息条件的定位','本轮出版社摘要/亮点','RSS及跨相关指纹路线；不等价无环境指纹的IQ联合拟合。'),
 ('A2LTFEUM','D 求解思想/基础理论','本轮PDF第1–2页模型与方法概述','单源LOS时延DPD，距离潜变量EM；不应把名称EM自动视为多径分离EM。'),
 ('10.1109/29.1552','D 求解思想/基础理论','作者PDF第1–2页§I–III定向核读','叠加分量分配与逐分量估计已有EM基础；不是新思想。'),
 ('10.1109/78.324732','D 求解思想/基础理论','作者页面/摘要；本地下载证书失败','SAGE小隐藏空间顺序更新；完整推导本轮未读，不声称复现。'),
 ('JUHLJTTN','D 求解思想/基础理论','作者稿引言定向核读','MESSL依赖时频点单源主导；当前完全重叠BPSK不具有已验证的W-disjoint条件。'),
 ('ZIR6J5Y9','D 求解思想/基础理论','复用已有全文记录','压缩多通道盲解卷积可辨识性；不是双源全局位置唯一性的现成定理。'),
 ('10.1109/jsait.2023.3287823','D 求解思想/基础理论','作者稿PDF第5页§II及式(20)–(22)核读','两个未知波形＋未知稀疏信道已有研究，但要求已知低维随机表示基；不直接等价。'),
 ('10.1016/j.sigpro.2024.109484','D 求解思想/基础理论','作者稿PDF第7页§2.4核读','上一项多天线SoMAN扩展；已知T/D表示矩阵，不可默认为当前公共带限先验。'),
 ('10.1109/tit.2025.3566885','D 求解思想/基础理论','本轮元数据＋作者摘要；全文待核','离网盲消息恢复/用户解混；方法家族近邻，未确认其基结构能否满足当前模型。'),
 ('10.1016/j.dsp.2026.106007','D 求解思想/基础理论','本轮正式元数据＋作者摘要','低维PSF/点源、低秩盲超分辨解混；不把其收敛定理迁移成定位保证。'),
 ('10.1016/s0165-1684(99)00027-4','D 求解思想/基础理论','作者/出版社摘要及作者论文章节','单通道二阶统计多径参数可辨识；仅限条件提醒，不是当前双源定位不可辨识证明。'),
 ('10.1049/rsn2.12408','D 求解思想/基础理论','本轮出版社式(8)–(9)','每记录一源的l0=1分配约束；不能直接代替同时混叠双源的联合拟合。'),
]


def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def main():
    library = json.loads((OUT/'zotero_inventory.json').read_text(encoding='utf-8'))
    by_key = {r['key']: r['data'] for r in library}
    by_doi = {d.get('DOI','').lower(): d for d in by_key.values() if d.get('DOI')}
    meta = {}
    for fn in ['search_raw.json', 'search_extended.json', 'nearest_metadata.json', 'last_branch.json']:
        obj = json.loads((OUT/fn).read_text(encoding='utf-8'))
        for r in obj.get('results', obj).values():
            if not isinstance(r,dict):
                continue
            msg = r.get('data',{}).get('message',{})
            for w in msg.get('items', [msg]):
                if w.get('DOI'):
                    meta[w['DOI'].lower()] = w
    rows = []
    for ident, family, level, note in ROWS:
        d = by_key.get(ident, by_doi.get(ident.lower(), {}))
        m = meta.get(ident.lower(), {})
        title = d.get('title') or (m.get('title') or [''])[0]
        assert title, ident
        if not d:
            d = next((x for x in by_key.values() if norm(x.get('title',''))==norm(title)), {})
        doi = d.get('DOI') or m.get('DOI','')
        url = 'https://doi.org/'+doi if doi else d.get('url','')
        assert url, ident
        date = d.get('date') or str(m.get('published',{}).get('date-parts',[['']])[0][0])
        venue = d.get('publicationTitle') or d.get('proceedingsTitle') or (m.get('container-title') or [''])[0]
        rows.append(dict(id=f'N{len(rows)+1:02}', title=title, doi=doi, url=url, date=date,
            venue=venue, zotero_key=d.get('key',''), family=family, evidence=level, implication=note))
    (OUT/'neighbor_matrix.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2),encoding='utf-8')
    lines=['# 近邻文献分层矩阵','', '检索日期：2026-10-30。A=核心场景/联合框架；B=关联初始化；C=信息条件不同；D=思想与理论。分层不是重要性排名，也不表示全部全文精读。Zotero只读，未改分类或阅读状态。', '']
    for family in dict.fromkeys(r['family'] for r in rows):
        lines += ['## '+family,'','| 编号与文献 | 来源/库内键 | 本轮证据范围 | 与当前问题的关系 |','|---|---|---|---|']
        for r in rows:
            if r['family']==family:
                lines.append(f"| {r['id']} [{r['title']}]({r['url']}) | {r['venue']}；{r['date']}；{r['zotero_key'] or '未在库中匹配'} | {r['evidence']} | {r['implication']} |")
        lines.append('')
    (OUT/'近邻文献矩阵.md').write_text('\n'.join(lines),encoding='utf-8')
    missing=[r for r in rows if not r['zotero_key']]
    lines=['# 本轮尚未入库的相关文献','', '以下已按895条Zotero顶层记录匹配DOI和规范化标题。优先给正式发表版本；已得到作者稿的也列出，供统一归档。并非每篇都必须补全文，更不需要全部复现。', '', '## 建议优先归档的方法近邻与理论来源','']
    for r in missing:
        if r['family'].startswith('D') or r['doi'].lower()=='10.1016/j.sigpro.2020.107907':
            lines += [f"- **{r['id']} {r['title']}** — {r['venue']}，{r['date']}。[正式页面]({r['url']})。{r['evidence']}。{r['implication']}"]
    lines += ['', '## 信息条件不同的补充文献（不阻塞当前主线）','']
    for r in missing:
        if not(r['family'].startswith('D') or r['doi'].lower()=='10.1016/j.sigpro.2020.107907'):
            lines += [f"- **{r['id']} {r['title']}** — {r['venue']}，{r['date']}。[正式页面]({r['url']})。{r['evidence']}。{r['implication']}"]
    lines += ['', '## 已有可读作者稿，不必为等待出版社PDF暂停研究','',
      '- DBD 2023：[作者稿](https://arxiv.org/abs/2208.04381)，本轮保存并核读模型条件。',
      '- 多天线DBD 2024：[作者稿](https://arxiv.org/abs/2303.13609)，本轮保存并核读§2.4。',
      '- Pseudo-ML 2021：[作者稿](https://arxiv.org/abs/1808.00857)，本轮保存并核对已知训练符号前提。',
      '- Feder–Weinstein 1988：[作者PDF](https://www.eng.tau.ac.il/~meir/articles/3%20Parameter%20Estimation%20of%20Superimposed%20Signals%20Using%20the%20EM.pdf)。',
      '- SAGE 1994：[作者稿链接](https://web.eecs.umich.edu/~fessler/papers/files/jour/94/pre/tsp%2Csage.pdf)。网页可读，本地下载证书校验失败；没有关闭证书验证。',
      '- 盲用户解混2025：[作者稿入口](https://arxiv.org/abs/2406.17393)，本轮未全文精读。',
      '- 快速盲超分辨/解混2026：[作者稿入口](https://arxiv.org/abs/2407.09900)，本轮未全文精读。',
      '', 'Friedlander2014已入库KGA2WLPL/47IHA9KL并完成本轮全文阅读，不再列为待补。其引用[4]标为in preparation的长稿，尚未确认正式发表身份；不提供猜测DOI。']
    (OUT/'未入库文献与链接.md').write_text('\n'.join(lines),encoding='utf-8')
    print('matrix',len(rows),'missing',len(missing))


if __name__=='__main__':
    main()
