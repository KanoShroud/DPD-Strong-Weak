"""Second-pass metadata verification and lawful author-copy retrieval."""
from concurrent.futures import ThreadPoolExecutor
from collect_evidence import ROOT, metadata, pdf, write_json

DOIS = [
    '10.1109/TASL.2008.2004533', '10.1109/TVT.2020.2968136',
    '10.1109/TSP.2022.3215651', '10.1109/ICCWorkshops67674.2025.11162218',
    '10.1016/j.sigpro.2025.110040', '10.1109/TAP.2019.2891403',
    '10.1109/TWC.2026.3722704',
]
PDFS = {
    'mbd2020_arxiv': 'https://arxiv.org/pdf/2001.00613',
    'semiblind2022_mit': 'https://sia.mit.edu/wp-content/uploads/2022/06/2022-weiss-arikan-vishnu-deane-singer-wornell-tsp.pdf',
    'mpdpd2018': 'https://mdpi-res.com/d_attachment/sensors/sensors-18-00892/article_deploy/sensors-18-00892.pdf',
    'dra2021': 'https://sgvr.kaist.ac.kr/~ikan/papers/DRA_SSL/Diffraction-_and_Reflection-Aware_Multiple_Sound_Source_Localization.pdf',
    'passive_structure2023': 'https://arl.nus.edu.sg/wp-content/uploads/2025/03/An-architecture-for-passive-joint-localization-and-structure.pdf',
}

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        metas = list(pool.map(metadata, DOIS))
        sources = list(pool.map(pdf, PDFS.items()))
    write_json(ROOT/'crossref_followup.json', metas)
    write_json(ROOT/'source_manifest_followup.json', sources)
    for m in metas:
        print(m['doi'], m['ok'], m.get('metadata',{}).get('title'))
    for s in sources:
        print(s)
