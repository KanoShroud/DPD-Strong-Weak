"""几何/波形/噪声按角色和编号确定；仅强源功率随gap改变。"""
import numpy as np

from ..g8.physics import FS, C, RECEIVERS
from ..g8.scenes import rrc, RATE, BW

SEED = 2026101009


def make(role, group, realization, overlap, snr_db, gap_db, n=16384):
    if role not in ('dev', 'check', 'smoke'):
        raise ValueError('没有test数据入口')
    role_id = {'dev': 0, 'check': 1, 'smoke': 2}[role]
    geo = np.random.default_rng(np.random.SeedSequence([SEED, role_id, group, 0]))
    while True:
        pos = geo.uniform(-1000, 1000, (2, 2))
        distance = np.linalg.norm(pos[:, None]-RECEIVERS, axis=-1)
        if distance.min() >= 150 and np.linalg.norm(pos[0]-pos[1]) >= 300:
            break
    rng = np.random.default_rng(np.random.SeedSequence([SEED, role_id, group, realization, 1]))
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, role_id, group, realization, 2]))
    center = rng.uniform(-FS/2+BW, FS/2-BW)
    shift = BW*(1-overlap)/(1+overlap)
    centers = center+np.array([-.5, .5])*shift
    if rng.random() < .5:
        centers = centers[::-1]
    pulse_energy = np.sum(rrc(np.arange(-10, 10, .001))**2)*.001
    signal = np.empty((2, 4, n), complex)
    gain = 1/distance**2
    weak_power = (BW/FS)*10**(snr_db/10)
    tx = weak_power*np.array([10**(gap_db/10), 1.])/gain.mean(-1)
    for source in range(2):
        bits = rng.choice([-1., 1.], size=int(n*RATE/FS)+512)
        timing, phase = rng.random(), rng.uniform(-np.pi, np.pi)
        for station in range(4):
            times = np.arange(n)/FS-distance[source, station]/C
            u = times*RATE-timing
            base = np.floor(u).astype(int)
            value = np.zeros(n)
            for delta in range(-10, 11):
                symbol = base+delta
                value += bits[symbol+256]*rrc(u-symbol)
            signal[source, station] = (value/np.sqrt(pulse_energy)
                * np.exp(2j*np.pi*centers[source]*times+1j*phase)
                * np.sqrt(tx[source]*gain[source, station]))
    noise = (noise_rng.normal(size=(4, n))+1j*noise_rng.normal(size=(4, n)))/np.sqrt(2)
    return dict(positions=pos, signal=signal, noise=noise)
