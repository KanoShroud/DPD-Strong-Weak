"""Build an import recommendation, never write to Zotero."""
from pathlib import Path
import json
import re
from difflib import SequenceMatcher
from collect_evidence import write_json

ROOT = Path(__file__).resolve().parent
ENTRIES = [
 ('A1','10.1109/TWC.2025.3554614','Cooperative Direct Localization in Multipath Environments Using Binary Sparse Modeling and Cayley–Menger Determinant','TWC 2025','03 结构近邻','仅书目/摘要线索；必须补全文','反射几何消歧与二元共同支持的最近后续'),
 ('A2','10.1109/TAES.2024.3454564','Joint Localization and Source Association Sparse Bayesian Learning Under Multipath Propagation','TAES 2025','02 关联与链路缺失','仅书目/摘要线索；必须补全文','多源关联、相干多径与衰减系数联合估计'),
 ('A3','10.1109/TASL.2008.2004533','Disambiguation of TDOA Estimation for Multiple Sources in Reverberant Environments','TASLP 2008','02 关联与链路缺失','原文§II、IV–VII定向核验','不同源缺失不同边、无需共同参考、虚假一致性'),
 ('A4','10.1109/TSP.2016.2593690','TDOA Matrices: Algebraic Properties and Their Application to Robust Denoising With Missing Data','TSP 2016','02 关联与链路缺失','作者稿摘要、§II、VI–VIII缺失/异常模型核验','不能将低秩补全与剔除异常本身称为新贡献'),
 ('A5','10.1109/TSP.2020.3011804','Identifiability Conditions for Compressive Multichannel Blind Deconvolution','TSP 2020','04 可辨识性','作者稿§II–IV条件核验；正式版待导入','单输入盲解卷积条件不能直接移用多源'),
 ('A6','10.1109/TSP.2022.3173731','A Semi-Blind Method for Localization of Underwater Acoustic Sources','TSP 2022','04 可辨识性','正式原文§II–IV及VII-B/C核验','无LOS可解依赖几何；已经讨论多源扩展'),
 ('A7','10.1109/ACSSC.2014.7094537','Direct localization of emitters using widely spaced sensors in multipath environments','Asilomar 2014','03 结构近邻','会议书目/摘要；详方法参照相关作者稿A8','已知波形多源、部分LOS阻塞的正式近邻'),
 ('A8','','High Precision TOA-based Direct Localization of Multiple Sources in Multipath','arXiv 2015','03 结构近邻','作者稿§II–IV定向核验','理论保证与原子范数；不冒充已确认期刊版本'),
 ('A9','10.3390/s18030892','Direct Position Determination of Unknown Signals in the Presence of Multipath Propagation','Sensors 2018','01 盲多源时延定位','全文HTML模型/算法章节及PDF模型核验','未知多源已有；已知转发器/反射位置、阵列、非负实衰减'),
 ('B1','10.1109/TSP.2022.3215651','Location Estimates From Channel State Information via Binary Programming','TSP 2022','03 结构近邻','IEEE正式摘要/元数据；全文待读','二元支持转换条件及字典设计的原始出处'),
 ('B2','10.1109/TSP.2017.2666779','Direct Localization for Massive MIMO','TSP 2017','03 结构近邻','作者机构摘要/元数据；本轮未阅读全文','DiSouL共同位置/站特异反射结构'),
 ('B3','10.1109/ACCESS.2018.2889958','Unified Subspace Fitting Framework and Its Performance Analysis for Direct Position Determination in the Presence of Multipath Propagation','IEEE Access 2019','01 盲多源时延定位','书目与作者稿检索片段；全文待读','2018 MP-DPD后续，不能忽略统一子空间框架'),
 ('B4','10.1109/TWC.2025.3601543','Decentralized Indoor Direct Localization With Multiple Wi-Fi Access Points','TWC 2026（2025在线）','03 结构近邻','正式PDF§II–III定向核验','多径干扰参数+EM/AP；单用户ULA/CSI'),
 ('B5','10.3390/s22197245','A Direct Position Determination Method Based on Subspace Orthogonality in Cross-Spectra under Multipath Environments','Sensors 2022','01 盲多源时延定位','沿用前轮在线模型核验；本轮核对书目','Chai2025前身、参考/多径数权限'),
 ('B6','10.1109/TVT.2020.2968136','Propagation Measurements and Modeling for Low Altitude UAVs From 1 to 24 GHz','TVT 2020','00 低空实测与需求','正式版作者公开PDF§II及结论核验','遮挡、绕射/反射机制，不是定位算法验证'),
 ('B7','10.1016/j.dib.2025.112086','Air-to-ground channel dataset via UAV-aided measurements in multiple scenarios','Data in Brief 2025','00 低空实测与需求','出版商数据说明核验','公开CIR用于选传播参数，不是盲多站混合IQ'),
 ('C1','10.1121/10.0016999','An architecture for passive joint localization and structure learning in reverberant environments','JASA 2023','90 不同信息条件','原文引言/架构范围核验；未全篇','利用运动和环境学习；不直接成为静态快照方案'),
 ('C2','10.1109/TRO.2021.3118966','Diffraction- and Reflection-Aware Multiple Sound Source Localization','TRO 2022（2021在线）','90 不同信息条件','IEEE摘要与作者稿引言核验','已重建三维环境、声线和多源；若加地图必须比较'),
 ('C3','','A graph theoretical framework for consistent time differences of arrival','ITG Speech Communication 2010','02 关联与链路缺失','作者机构会议稿摘要/图结构核验','DATEMM后续；无共同参考和不完整图已有理论'),
]

def norm(t):
    return re.sub(r'[^a-z0-9]+','', t.lower())

def main():
    items=json.loads((ROOT/'zotero_inventory.json').read_text(encoding='utf-8'))
    known={}
    for path in ROOT.glob('crossref*.json'):
        for row in json.loads(path.read_text(encoding='utf-8')):
            if row.get('ok'):
                known[row['doi'].lower()]=row['metadata']
    entries=[]
    for code,doi,title,venue,category,read,reason in ENTRIES:
        hits=[]
        for item in items:
            d=item.get('data',{})
            if d.get('itemType') in ['attachment','note']:
                continue
            exact_doi=bool(doi) and d.get('DOI','').lower()==doi.lower()
            title_match=SequenceMatcher(None,norm(title),norm(d.get('title',''))).ratio()>=0.97
            if exact_doi or title_match:
                children=[c['key'] for c in items if c.get('data',{}).get('parentItem')==item['key'] and c.get('data',{}).get('contentType')=='application/pdf']
                hits.append({'key':item['key'],'title':d.get('title'),'pdfs':children})
        link='https://doi.org/'+doi if doi else ('https://arxiv.org/abs/1505.03193' if code=='A8' else 'https://www.iss.uni-stuttgart.de/forschung/publikationen/kreissig_itg10.pdf')
        entries.append(dict(id=code,doi=doi,title=title,venue=venue,category=category,reading=read,reason=reason,url=link,zotero=hits,crossref_verified=doi.lower() in known))
    write_json(ROOT/'reference_recommendations.json',entries)
    lines=['# Zotero待导入与补附件清单','', '核对日期：2026-10-25。按DOI与规范题名核对本地库快照；未执行导入、改分类或元数据更新。正式版本优先。A8为独立保留的相关扩展作者稿，不能用A7会议版替代其全部定理，也未确认其同题期刊版。','', '|优先级|文献与版本|库内状态|阅读范围|需要它回答什么|建议分类|','|---|---|---|---|---|---|']
    for r in entries:
        status='未检出：新增条目'
        if r['zotero']:
            status='；'.join(f"已有 {h['key']}："+('有PDF，不重复导入' if h['pdfs'] else '仅补PDF，不新增条目') for h in r['zotero'])
        lines.append(f"|{r['id']}|[{r['title']}]({r['url']})；{r['venue']}|{status}|{r['reading']}|{r['reason']}|{r['category']}|")
    lines.extend(['','## 附件与版本提醒','','- 已有TAES条目W8HIR95K保留；正式卷期为2025年61(1):1104–1119，旧记录2024为在线/旧元数据，补PDF后再核对，不创建重复项。','- Briso的arXiv上传时间为2025，但PDF和IEEE均为2020年TVT正式论文，按2020登记。','- Weiss2022的SBL是semi-blind localization，不是sparse Bayesian learning。','- TWC2026 DIM-DPD的2025在线日期与2026卷年不冲突。','- 当前已经读到网上全文，也仍应导入库内留档；本轮PDF副本见sources目录，不自动写Zotero。'])
    (ROOT/'待导入文献与分类清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'entries':len(entries),'not_in_library':sum(not r['zotero'] for r in entries),'existing':[(r['id'],r['zotero']) for r in entries if r['zotero']]},ensure_ascii=False))

if __name__=='__main__':
    main()
