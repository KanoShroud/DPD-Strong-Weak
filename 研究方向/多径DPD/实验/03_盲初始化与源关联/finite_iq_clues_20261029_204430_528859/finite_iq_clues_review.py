"""Saved-IQ audit and labelled component attribution, no localization solver."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from finite_iq_clues import FS, N, B, C, UPSAMPLE, SOURCE, STATIONS, PAIRS, sha, signature, write, peak_candidates


def attribution(out):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    answers = []
    for rep, ax in zip([0, 5], axes):
        z = np.load(out/f'L1_R{rep}_components.npz')
        raw = np.load(SOURCE/f'L1_R{rep}_P2.npz')
        m, n = 0, 3
        coeff = raw['g'][n, :, 0]*raw['g'][m, :, 0].conj()
        length = 2*N
        spectrum = fft(z['los'], n=length, axis=2)
        same = spectrum[:, n]*spectrum[:, m].conj()
        mixed = spectrum[:, n].sum(axis=0)*spectrum[:, m].sum(axis=0).conj()
        aligned = same[0]+same[1]*np.exp(1j*np.angle(coeff[0]/coeff[1]))
        csds = dict(mixture=mixed, same_source_sum=same.sum(axis=0),
                    aligned_oracle=aligned, source_a=same[0], source_b=same[1])
        mask = abs(fftfreq(length,1/FS)) <= B/2
        ll = length*UPSAMPLE
        lag = np.arange(-ll//2,ll//2)/(FS*UPSAMPLE)
        bound = np.linalg.norm(STATIONS[n]-STATIONS[m])/C
        keep = abs(lag) <= bound
        lag = lag[keep]
        curves = {}
        for name, csd in csds.items():
            pad = np.pad(fftshift(csd*mask), ((ll-length)//2,)*2)
            curves[name] = fftshift(ifft(ifftshift(pad)))[keep]*UPSAMPLE/N
        ranges = np.linalg.norm(STATIONS[:,None]-z['truth'][None],axis=2)
        target = (ranges[n]-ranges[m])/C
        estimates = {}
        for name, curve in curves.items():
            peaks = peak_candidates(lag, abs(curve))
            estimates[name] = dict(nearest_m=np.min(abs(np.array(peaks['lag_s'])[:,None]-target[None]),axis=0)*C,
                                   peak_lag_s=peaks['lag_s'])
        midpoint = target.mean()
        local = abs(lag-midpoint) <= 3/B
        norm = np.max(abs(curves['source_a'][local])+abs(curves['source_b'][local]))
        for name, title, color, ls in [('mixture','实际双源相关模','#0072B2','-'),
                                     ('same_source_sum','去掉有限样本跨源项','#D55E00','--'),
                                     ('aligned_oracle','相关贡献相位对齐（诊断）','#009E73',':')]:
            ax.plot((lag[local]-midpoint)*C,abs(curves[name][local])/norm,label=title,color=color,ls=ls,lw=1.8)
        for q in range(2):
            ax.axvline((target[q]-midpoint)*C,color='#666666',ls=['-.','--'][q],lw=1)
        ax.set_title(f'构型1、实现{rep}：直达、无噪声\n相关贡献相位差 {np.angle(coeff[1]/coeff[0],deg=True):.1f}°')
        ax.set_xlabel('相对两源真实时差中点的距离差（m）')
        ax.set_ylabel('相关模 / 共同幅度尺度')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        note = dict(realization=rep, station_pair=[m,n],
                    direct_delay_difference_m=float(abs(target[1]-target[0])*C),
                    relative_phase_deg=float(np.angle(coeff[1]/coeff[0],deg=True)),
                    coefficient_ratio=float(abs(coeff[1]/coeff[0])),
                    coincident_delay_cancellation_factor=float(abs(sum(coeff))/sum(abs(coeff))),
                    estimates=estimates, role='truth-component attribution, not implementable preprocessing')
        answers.append(note)
        np.savez_compressed(out/f'attribution_L1_R{rep}.npz', lag_s=lag,target_s=target,
                            coeff=coeff,**curves)
    fig.suptitle('复相关贡献可以相消；独立发射源不等于相关贡献不会叠加',fontsize=13)
    fig.savefig(out/'相关峰偏移归因.png',dpi=180)
    fig.savefig(out/'相关峰偏移归因.pdf')
    plt.close(fig)
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--prefix',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve(strict=True)
    r = json.loads((out/'result.json').read_text(encoding='utf-8'))
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    before = {str(p):sha(p) for p in [out/'result.json',out/'protocol.json']+list(out.glob('*_components.npz'))}
    checks = {p:sha(p)==h for p,h in protocol['hashes'].items()}
    summaries = []
    for paths in ['LOS','MP']:
        for signal in ['A','B','AB']:
            for noise in ['clean','snr6']:
                rows = [x for x in r['rows'] if x['paths']==paths and x['signal']==signal and x['noise']==noise]
                for method in ['CC_ABS','PHAT_ABS']:
                    for count in [2,6,18]:
                        fields = [x['metrics'][method][str(count)] for x in rows]
                        err = np.array([f['nearest_lag_error_m'] for f in fields])
                        summaries.append(dict(paths=paths,signal=signal,noise=noise,method=method,count=count,n=len(rows),
                                              nearest_lag_rmse_m=float(np.sqrt(np.mean(err**2))),
                                              max_nearest_lag_error_m=float(err.max()),
                                              reference_complete=sum(f['tolerances']['0.25']['all_reference_sources'] for f in fields),
                                              all_pairs_complete=sum(f['tolerances']['0.25']['all_pairs_sources'] for f in fields),
                                              pair_source_coverage=float(np.mean([f['tolerances']['0.25']['pair_source_coverage'] for f in fields]))))
    for row in r['rows']:
        base=f'L{row["layout"]}_R{row["realization"]}'
        z=np.load(out/(base+'_components.npz'))
        source=z['los']+(z['reflected'] if row['paths']=='MP' else 0)
        iq=source[row['active_sources']].sum(axis=0)+(z['noise'] if row['noise']=='snr6' else 0)
        checks[row['tag']+'/identity']=signature(iq)==row['iq_sha']
        ranges=np.linalg.norm(STATIONS[:,None]-z['truth'][None],axis=2)
        for method in row['extraction']:
            computed=[]
            for m,n in PAIRS:
                candidates=np.array(row['extraction'][method][f'{m}_{n}']['lag_s'][:6])
                expected=(ranges[n,row['active_sources']]-ranges[m,row['active_sources']])/C
                computed.append(np.min(abs(candidates[:,None]-expected[None]),axis=0)*C)
            checks[row['tag']+'/'+method]=bool(np.array_equal(computed,row['metrics'][method]['6']['nearest_lag_error_m']))
    prefix=json.loads((args.prefix/'result.json').read_text(encoding='utf-8'))
    bytag={x['tag']:x for x in r['rows']}
    for row in prefix['rows']:
        checks['prefix/'+row['tag']]=row['iq_sha']==bytag[row['tag']]['iq_sha'] and row['metrics']==bytag[row['tag']]['metrics']
    diagnostic=attribution(out)
    checks.update({'unchanged/'+p:sha(p)==h for p,h in before.items()})
    assert all(checks.values()),[k for k,v in checks.items() if not v]
    write(out/'review.json',dict(summary=summaries,attribution=diagnostic,checks=checks,
                                  execution_seconds=r['seconds'], completed=r['completed'],
                                  sources=before, review_source_sha=sha(Path(__file__))))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print('review checks',len(checks),'PASS')
    print(json.dumps(diagnostic,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
