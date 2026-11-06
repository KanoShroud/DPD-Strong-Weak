"""Data-only complex correlation and consistent TDOA graph initializer.

Not full DATEMM; not a complete shared-reflector DPD estimator.
"""
import itertools

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

C=299792458.


def correlations(y,fs,stations,bandwidth,passband,mode):
    n=y.shape[1]
    nfft=2**int(np.ceil(np.log2(2*n-1)))
    up=8
    X=np.fft.fft(y,nfft,axis=1)
    f=np.fft.fftfreq(nfft,1/fs)
    mask=abs(f)<=passband
    output={}
    for i,j in itertools.combinations(range(len(stations)),2):
        spec=X[i]*X[j].conj()
        if mode=='phat':
            floor=1e-3*np.median(abs(spec[mask]))
            spec=spec/np.maximum(abs(spec),floor)
        spec*=mask
        centered=np.fft.fftshift(spec)
        expanded=np.pad(centered,(nfft*(up-1)//2,nfft*(up-1)//2))
        corr=abs(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(expanded))))
        lags=(np.arange(nfft*up)-nfft*up//2)/(fs*up)
        limit=np.linalg.norm(stations[i]-stations[j])/C
        keep=abs(lags)<=limit
        curve=corr[keep];lags=lags[keep]
        ids,_=find_peaks(curve,distance=max(1,int(fs*up/(2*bandwidth))))
        peaks=[]
        for k in ids:
            a,b,c=curve[k-1:k+2]
            denominator=a-2*b+c
            offset=.5*(a-c)/denominator if abs(denominator)>1e-30 else 0.
            offset=float(np.clip(offset,-.5,.5))
            peaks.append({'range_difference_m':float((lags[k]+offset/(fs*up))*C),
                          'score':float(curve[k]/max(np.max(curve),1e-30))})
        peaks.sort(key=lambda p:p['score'],reverse=True)
        output[f'{i}_{j}']={'all_peaks':peaks,'peaks':peaks[:16]}
    return output


def geometry_roots(stations,relative_lengths,height=None):
    assert np.ptp(stations[:,2])<1e-8
    delta=stations[1:,:2]-stations[0,:2]
    d=relative_lengths[1:]-relative_lengths[0]
    rhs=np.sum(delta**2,axis=1)-d**2
    if height is None:
        matrix=np.column_stack([2*delta,2*d])
        sol,_,rank,_=np.linalg.lstsq(matrix,rhs,rcond=1e-12)
        if rank<3:
            return []
        h2=sol[2]**2-sol[:2]@sol[:2]
        if h2<=0 or sol[2]<=0:
            return []
        return [np.r_[stations[0,:2]+sol[:2],stations[0,2]+np.sqrt(h2)]]
    uv=np.linalg.lstsq(2*delta,np.column_stack([rhs,-2*d]),rcond=1e-12)[0]
    if np.linalg.matrix_rank(delta)<2:
        return []
    u,v=uv[:,0],uv[:,1]
    poly=np.array([1-v@v,-2*u@v,-u@u-(height-stations[0,2])**2])
    roots=np.roots(np.trim_zeros(np.where(abs(poly)<1e-12,0.,poly),'f'))
    answers=[]
    for r in roots:
        if abs(r.imag)>1e-6 or r.real<=0:
            continue
        p=np.r_[stations[0,:2]+u+v*r.real,height]
        ranges=np.linalg.norm(stations-p,axis=1)
        if np.max(abs(ranges[1:]-ranges[0]-d))<1e-5:
            answers.append(p)
    return answers


def graphs_and_positions(pair_data,stations,tolerance_m,bounds):
    graphs=[];points=[]
    low,high=np.array(bounds[0]),np.array(bounds[1])
    def inside(p):
        return bool(np.all(p>=low)&np.all(p<=high))
    for size in [3,4]:
        for nodes in itertools.combinations(range(len(stations)),size):
            ref=nodes[0]
            choices=[pair_data[f'{ref}_{m}']['peaks'] for m in nodes[1:]]
            for star in itertools.product(*choices):
                relative=np.r_[0.,[-p['range_difference_m'] for p in star]]
                edges=[];scores=[];valid=True
                for a,b in itertools.combinations(range(size),2):
                    pair=pair_data[f'{nodes[a]}_{nodes[b]}']['peaks']
                    prediction=relative[a]-relative[b]
                    if not pair:
                        valid=False;break
                    peak=min(pair,key=lambda p:abs(p['range_difference_m']-prediction))
                    if abs(peak['range_difference_m']-prediction)>tolerance_m:
                        valid=False;break
                    edges.append((a,b,peak['range_difference_m']));scores.append(peak['score'])
                if not valid:
                    continue
                matrix=[];values=[]
                for a,b,value in edges:
                    row=np.zeros(size);row[a]=1;row[b]=-1
                    matrix.append(row[1:]);values.append(value)
                matrix=np.array(matrix);values=np.array(values)
                rel=np.r_[0.,np.linalg.lstsq(matrix,values,rcond=None)[0]]
                closure=float(np.max(abs(matrix@rel[1:]-values)))
                graph_id=len(graphs)
                graphs.append({'nodes':nodes,'relative_lengths_m':rel.tolist(),'closure_error_m':closure,
                               'score':float(np.mean(scores)),'points':0})
                heights=[None] if size==4 else np.arange(low[2],high[2]+.1,5.)
                for height in heights:
                    for p in geometry_roots(stations[list(nodes)],rel,height):
                        if not inside(p):
                            continue
                        if size==4:
                            pairs=list(itertools.combinations(range(size),2))
                            anchors=stations[list(nodes)]
                            def residual(x):
                                lengths=np.linalg.norm(anchors-x,axis=1)
                                return np.array([lengths[a]-lengths[b] for a,b in pairs])-values
                            fitted=least_squares(residual,p,bounds=(low,high),gtol=1e-8,xtol=1e-10,ftol=1e-10,max_nfev=80)
                            p=fitted.x
                            fit_error=float(np.max(abs(residual(p))))
                        else:
                            fit_error=closure
                        if fit_error>tolerance_m:
                            continue
                        graphs[-1]['points']+=1
                        points.append({'position_m':p.tolist(),'graph_id':graph_id,'nodes':nodes,'size':size,
                                       'score':float(np.mean(scores)),'fit_residual_m':fit_error})
    points.sort(key=lambda p:(-p['size'],-p['score'],p['fit_residual_m']))
    return {'graphs':graphs,'candidate_points':points}


def solve(mixture,public):
    stations=np.array(public['stations_m'])
    output={}
    for mode in ['ordinary','phat']:
        pair_data=correlations(mixture,public['fs_hz'],stations,public['bandwidth_hz'],public['passband_hz'],mode)
        versions={}
        for denominator in [4,2]:
            tol=C/(denominator*public['bandwidth_hz'])
            versions[str(denominator)]=graphs_and_positions(pair_data,stations,tol,public['bounds_m'])
        output[mode]={'pairs':pair_data,'versions':versions}
    return output


def unit_tests():
    rng=np.random.default_rng(20261025)
    signal=rng.normal(size=1024)+1j*rng.normal(size=1024)
    y=np.array([signal,np.r_[np.zeros(12),signal[:-12]]])
    stations=np.array([[0.,0.,4.],[200.,0.,4.]])
    pair=correlations(y,50e6,stations,12.5e6,7e6,'ordinary')['0_1']['peaks'][0]
    shifted=correlations(y*np.exp(1j*np.array([.8,-.4]))[:,None],50e6,stations,12.5e6,7e6,'ordinary')['0_1']['peaks'][0]
    anchors=np.array([[-400.,-300.,4.],[450.,-250.,4.],[380.,400.,4.],[-350.,350.,4.]])
    p=np.array([51.,-91.,65.]);d=np.linalg.norm(anchors-p,axis=1)
    root=geometry_roots(anchors,d-d[0])[0]
    curve=geometry_roots(anchors[:3],d[:3]-d[0],p[2])
    return {'correlation_sign':abs(pair['range_difference_m']+12/50e6*C)<1.,
            'complex_phase_invariance':abs(pair['range_difference_m']-shifted['range_difference_m'])<1e-8,
            'four_station_geometry':np.linalg.norm(root-p)<1e-6,
            'three_station_curve':min(np.linalg.norm(x-p) for x in curve)<1e-6}
