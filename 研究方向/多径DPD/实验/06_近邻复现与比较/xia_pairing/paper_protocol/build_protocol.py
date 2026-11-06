"""Freeze verified paper settings, show source figures; never run guessed sims."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import shutil
import urllib.request
import fitz

ROOT = Path(__file__).resolve().parent


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    pdf = next(Path('F:/Zotero/storage/USPFK577').glob('*.pdf')).resolve(strict=True)
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    doc = fitz.open(pdf)
    # Source figures only; no redraw, digitization, invented experimental points.
    crops = [('paper_fig2.png', 2, [300, 45, 600, 210]),
             ('paper_fig3.png', 3, [20, 45, 307, 205]),
             ('paper_snr.png', 2, [292, 200, 600, 301])]
    for name, page, bbox in crops:
        doc[page].get_pixmap(matrix=fitz.Matrix(3, 3), clip=fitz.Rect(bbox)).save(ROOT/name)
    url = 'https://api.crossref.org/works/10.1109%2FJSYST.2020.3005918'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Paper-reproduction-audit/1.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            metadata = json.load(response)
        write('crossref_raw.json', metadata)
        crossref = {'status': 'retrieved', 'endpoint': url,
                    'relation': metadata['message'].get('relation'),
                    'link': metadata['message'].get('link')}
    except Exception as exc:
        # Project Python's SSL store failed on this host. Use the system trust
        # store through PowerShell; never disable TLS certificate verification.
        try:
            command = ("[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
                       f"Invoke-RestMethod -Uri '{url}' -TimeoutSec 30 | ConvertTo-Json -Depth 100 -Compress")
            output = subprocess.run([shutil.which('pwsh') or 'pwsh', '-NoProfile', '-NonInteractive',
                                     '-Command', command], check=True, capture_output=True,
                                    text=True, encoding='utf-8', errors='strict', timeout=40)
            metadata = json.loads(output.stdout)
            write('crossref_raw.json', metadata)
            crossref = {'status': 'retrieved_system_tls', 'endpoint': url,
                        'python_ssl_error': str(exc), 'relation': metadata['message'].get('relation'),
                        'link': metadata['message'].get('link')}
        except Exception as fallback_error:
            crossref = {'status': 'unavailable', 'endpoint': url, 'error': str(exc),
                        'fallback_error': str(fallback_error)}
    loss = 42.6+20*math.log10(900.)
    config = {
        'status': 'SOURCE_DETAILS_REQUIRED_NO_NEW_LOCALIZATION',
        'source': {'doi': '10.1109/JSYST.2020.3005918', 'pdf': str(pdf), 'sha256': digest,
                   'pages_checked': [2855, 2856, 2857, 2858]},
        'verified': {'stations': 5, 'reference_m': [0, 0], 'region_side_m': 6000,
                     'modulation': 'BPSK', 'fc_hz': 900e6, 'fs_hz': 1e6, 'symbol_rate_hz': 200e3,
                     'samples': 2000, 'initial_particles': 1000,
                     'delta_h_m': 400, 'epsilon_printed': 0,
                     'resampling_covariance_m2': [[100, 0], [0, 100]],
                     'mc_trials': 1000, 'primary_metric': 'target j=1 RMSE',
                     'path_delays_us': [0, 3, 8, 11, 13, 21],
                     'relative_amplitudes_db': [0, -7, -15, -22, -24, -19],
                     'shadow_variance_db2': 25, 'figure2a_sir_db': 3,
                     'figure2a_snr_axis_db': [-10, -5, 0, 5, 10, 15, 20],
                     'figure2a_step_v_repetitions': 2,
                     'figure3a_sir_db': -3, 'figure3a_G': 3,
                     'figure3b_snr_db': 5, 'figure3b_sir_axis_db': [-2, 0, 2, 4, 6, 8, 10]},
        'unresolved': {k: None for k in ['noise_calibration_and_channel_normalization',
                                        'pulse_shaping', 'fractional_delay_and_fft_convention',
                                        'complex_channel_distribution', 'geometry_distribution',
                                        'source_label_association_and_missing_output',
                                        'G_path_count_convention', 'author_stop_rule']},
        'authorized_interpretation_not_original': 'previous <=0 stop and exact fifth-bin endpoint',
        'sanity_check_not_simulation': {'distance_km': 1, 'shadow_db': 0,
                                       'loss_db': loss, 'literal_tx_5db_to_rx_snr_db': 5-loss},
        'lookup': crossref,
        'source_search': {'queries': ['"10.1109/JSYST.2020.3005918" code',
                                     'full title + supplementary / Matlab / dataset',
                                     'DOI + correction'],
                          'result': 'No verified author code found in this bounded search',
                          'ieee_page': 'https://ieeexplore.ieee.org/document/9146280/',
                          'ieee_access': 'JavaScript verification; supplements not fully inspected'},
        'new_localization_solves': 0}
    write('原文参数合同.json', config)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Xia原文性能复现：设定与缺口</title>
<style>body{{max-width:1000px;margin:auto;padding:36px 22px;background:#f5f4ef;color:#203c46;font:16px/1.85 "Microsoft YaHei",system-ui}}section{{background:white;border:1px solid #dce6e1;border-radius:15px;padding:22px 28px;margin:20px 0}}h1{{font-size:32px;line-height:1.4}}h2{{font-size:23px}}.note{{background:#fff0d6;border-left:4px solid #be9444;padding:15px}}img{{width:100%;max-width:900px;display:block;margin:14px auto}}a{{color:#087b73}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;border-bottom:1px solid #e1e8e4;padding:9px}}.muted{{color:#627980}}summary{{cursor:pointer;font-weight:bold}}code{{font-size:14px}}</style>
<section class="note"><b>用户已授权按通行DSP做法补全，不再等待作者。</b><a href="../reconstructed/公开信息重建报告.html">最新固定约定与84次有噪短测 →</a>。以下为补全前的原文规格核对。</section>
<p class="muted">2026-10-27 · 历史规格核对阶段 · 此阶段新增定位实验0次</p><h1>回到作者的实验设置，<br>不再把自选控制当作原文性能</h1>
<section><p><b>已经完成：</b>逐项核对正文、表I、图2和图3，冻结明确参数，检索公开源码。</p><p><b>尚未完成：</b>独立复现作者的RMSE曲线。现有诊断不是原文完整复现，不据其失败断定Xia方法弱。</p><p class="note">关键阻塞不是算力，而是原文未明确的功率/噪声标定与生成、输出细节。此处不擅自填写默认值后运行。</p></section>
<section><h2>原文实际报告了什么？</h2><p>以下是用户提供PDF中的原图裁剪，不是我们重新仿真的结果。</p><img src="paper_fig2.png" alt="Xia2021图2原图：RMSE随SNR及迭代次数变化"><p>图2(a)：SIR=3dB，不同路径数；图2(b)：SNR=5dB、SIR=3dB，不同粒子数和迭代次数。原图显示RMSE随SNR和迭代改善，随后趋于稳定。</p>
<img src="paper_fig3.png" alt="Xia2021图3原图：方法比较"><p>图3(a)：SIR=−3dB、G=3；图3(b)：SNR=5dB、G=3。作者报告其方法优于所比较的方法。我们尚未独立验证这些曲线，也不从截图编造精确数值。</p><p class="muted">来源：IEEE Systems Journal 15(2), pp.2857—2858, Fig.2—3。目标指标为j=1的1000次Monte Carlo RMSE，不是我们后来使用的双源20m成功率。</p></section>
<section><h2>这些参数不再改动</h2><table><tr><th>项目</th><th>原文值</th></tr><tr><td>区域与接收站</td><td>6×6km²、5站、参考站在原点，其余随机部署</td></tr><tr><td>信号</td><td>BPSK、900MHz载频、1MHz采样、200kbps符号率、2000点</td></tr><tr><td>搜索</td><td>1000初始粒子、Δh400m、Γ=diag(100,100)m²</td></tr><tr><td>信道</td><td>COST231及5dB阴影标准差、表I路径延迟与相对幅度</td></tr><tr><td>统计</td><td>1000次Monte Carlo、目标源RMSE</td></tr></table><p>不加入Hann、当前±7MHz带限、RRC=0.25默认值、当前50MHz/8192点、额外选峰或联合精修。不用无噪单例代替原文有噪实验。</p></section>
<section><h2>最需要确认的一处：噪声到底加多大？</h2><img src="paper_snr.png" alt="原文式18和19及发射功率定义"><p>式(18)—(19)写发射功率与观测噪声之比，同时式(17)包含绝对路径损耗。以1km、900MHz、零阴影为例，损耗为<b>{loss:.2f}dB</b>。若直接按字面叠用，名义5dB会对应约<b>{5-loss:.2f}dB</b>的接收直达SNR。</p><p>这只是说明实现还需要明确归一化/标定步骤，并不证明论文错误。未经确认，不能自行改成“接收信号measured SNR”，再把跑出来的图称作原文复现。</p></section>
<section><h2>还缺哪些信息？</h2><p>最有效的是获得作者生成图2、图3的脚本，或一个最小仿真示例。次选是确认：</p><ol><li>路径损耗、发射功率、接收功率与噪声方差的归一化关系。</li><li>BPSK成形、非整数时延、复信道相位，以及源/站部署和随机重复方式。</li><li>ε=0时真实停止代码、空簇处理，以及目标j=1的关联和RMSE统计。</li></ol><p>有限网页检索和Crossref查询未找到可验证作者代码；IEEE页面受JavaScript验证限制，不能断言没有补充附件。</p></section>
<section><h2>接下来怎样执行才可靠？</h2><p>拿到关键说明 → 冻结有来源的配置 → 原图条件小批核对 → 1000次原尺度统计 → 才迁移到当前场景。不会为了跑通而修改原阈值或提高性能，也不会把未复现成功解释成我们创新的依据。</p><details><summary>给作者的简短核实问题</summary><p>能否分享图2、图3的仿真脚本，或说明功率/噪声标定、波形成形/时延、复路径系数和部署方式，以及ε=0和空簇的实际处理？本项目希望准确复现，而非根据自选补全评价论文。</p><p>此问题仅为草案，未发送邮件或联系作者。</p></details></section>
<p><a href="原文复现规格.md">完整规格与核实问题</a> · <a href="原文参数合同.json">已知/未定参数JSON</a> · <a href="https://doi.org/10.1109/JSYST.2020.3005918">原论文</a></p></html>'''
    (ROOT/'原文性能复现说明.html').write_text(html, encoding='utf-8')
    assert config['new_localization_solves'] == 0
    assert all(v is None for v in config['unresolved'].values())
    print(json.dumps({'out': str(ROOT), 'metadata': crossref['status'], 'new_solves': 0,
                      'pdf_sha256': digest, 'literal_loss_db': loss}, ensure_ascii=False))


if __name__ == '__main__':
    main()
