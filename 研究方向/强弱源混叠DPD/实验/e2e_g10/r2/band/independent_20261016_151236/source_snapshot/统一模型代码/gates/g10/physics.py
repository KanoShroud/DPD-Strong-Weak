"""Complex128 common model. All frequencies retained, no station normalization.

Reference: Chen/Hudson/Yao (2002) II.A/B, complex-data variable projection
adaptation. See method_sources.md for exact native/adapted boundaries.
"""
import numpy as np
import torch
from ..g8.physics import FS, C, RECEIVERS


def grid(step):
    axis = np.arange(-1000., 1000.001, step)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    return np.stack([x.ravel(), y.ravel()], -1)


def shortlist(points, scores, count, separation=30.):
    keep = []
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        if np.isfinite(scores[i]) and all(np.linalg.norm(points[i]-points[j]) >= separation for j in keep):
            keep.append(int(i))
            if len(keep) == count:
                break
    return keep


def atom(points, freq, stations, normalized=True):
    """Return [frequency, station, candidate]; d floor only inside 1 m of a station."""
    distance = torch.linalg.vector_norm(points[:, None]-stations, dim=-1).clamp_min(1.)
    a = torch.exp(-2j*torch.pi*freq[:, None, None]*distance.T[None]/C)/distance.T[None]
    return a/torch.linalg.vector_norm(a, dim=1, keepdim=True) if normalized else a


def fit(a, x):
    """SVD least-squares, rank deficient frequencies included rather than dropped."""
    s = (torch.linalg.pinv(a, rtol=1e-10) @ x[..., None]).squeeze(-1)
    return s, x-(a@s[..., None]).squeeze(-1)


class Model:
    def __init__(self, iq, check=lambda: None, device='cuda'):
        iq = np.asarray(iq, dtype=np.complex128)
        if iq.ndim != 2 or iq.shape[0] != 4 or not np.isfinite(iq).all():
            raise ValueError('IQ must be finite [4,N]')
        self.iq, self.n, self.check, self.device = iq, iq.shape[1], check, device
        self.stations = torch.as_tensor(RECEIVERS, dtype=torch.float64, device=device)
        self.f = torch.as_tensor(np.fft.fftfreq(self.n, 1/FS), device=device)
        self.x = torch.as_tensor(np.fft.fft(iq).T/self.n, device=device)
        self.energy = self.x.abs().square().sum()
        self.counts = dict(candidate_points=0, frequency_response_points=0,
                           joint_evaluations=0, covariance_decompositions=0, iterations=0)

    def response(self, positions, freq=None):
        p = positions if torch.is_tensor(positions) else torch.as_tensor(
            np.asarray(positions).reshape(-1, 2), dtype=torch.float64, device=self.device)
        f = self.f if freq is None else freq
        self.counts['frequency_response_points'] += len(p)*len(f)
        return atom(p, f, self.stations)

    def fitted(self, positions):
        a = self.response(positions)
        s, residual = fit(a, self.x)
        return a, s, residual

    def joint_tensor(self, p):
        _, residual = fit(self.response(p.reshape(-1, 2)), self.x)
        return residual.abs().square().sum()/self.energy

    def joint(self, positions, gradient=False):
        self.check()
        self.counts['joint_evaluations'] += 1
        p = torch.tensor(np.asarray(positions).reshape(-1, 2), dtype=torch.float64,
                         device=self.device, requires_grad=gradient)
        value = self.joint_tensor(p)
        if gradient:
            grad = torch.autograd.grad(value, p)[0]
            return float(value.detach().cpu()), grad.detach().cpu().numpy().ravel()
        return float(value.detach().cpu())

    def joint_batch(self, population, chunk=4):
        """Same SVD objective, small population batches to avoid scalar GPU launches."""
        values = []
        for lo in range(0, len(population), chunk):
            self.check()
            p = torch.as_tensor(np.asarray(population[lo:lo+chunk]), dtype=torch.float64, device=self.device)
            batch, count = p.shape[:2]
            a = self.response(p.reshape(-1, 2)).reshape(self.n, 4, batch, count).permute(2, 0, 1, 3)
            _, residual = fit(a, self.x.expand(batch, -1, -1))
            values.extend((residual.abs().square().sum((1, 2))/self.energy).cpu().tolist())
            self.counts['joint_evaluations'] += batch
        return np.asarray(values)

    def conditional(self, points, others=(), residual=None, chunk=64):
        """AP projects both candidate and data; NOMP supplies its raw fitted residual."""
        points = np.asarray(points).reshape(-1, 2)
        x, projector = self.x if residual is None else residual, None
        if len(others):
            a = self.response(others)
            projector = torch.eye(4, device=self.device)-a@torch.linalg.pinv(a, rtol=1e-10)
            x = (projector@x[..., None]).squeeze(-1)
        result = []
        with torch.no_grad():
            for i in range(0, len(points), chunk):
                self.check()
                aa = self.response(points[i:i+chunk])
                if projector is not None:
                    aa = projector@aa
                norm = aa.abs().square().sum(1)
                dot = torch.einsum('flq,fl->fq', aa.conj(), x)
                values = torch.where(norm > 1e-20, dot.abs().square()/norm.clamp_min(1e-20), 0.)
                result.extend((values.sum(0)/self.energy).cpu().tolist())
                self.counts['candidate_points'] += aa.shape[-1]
        return np.asarray(result)

    def residual_score(self, p, residual):
        aa = self.response(p.reshape(-1, 2)).squeeze(-1)
        dot = (aa.conj()*residual).sum(-1)
        return -dot.abs().square().sum()/self.energy


class Spectrum:
    """Known-response wideband MUSIC and incoherent Capon sum (not original HR)."""
    def __init__(self, model, method, count, segments=4, loading=1e-3):
        self.m, self.method = model, method
        if model.n % segments or segments < 4:
            raise ValueError('Covariance segments must divide N and be >=4')
        nf = model.n//segments
        data = torch.as_tensor(np.fft.fft(model.iq.reshape(4, segments, nf), axis=-1)/nf,
                               device=model.device).permute(2, 0, 1)
        self.cov = data@data.mH/segments
        self.freq = torch.as_tensor(np.fft.fftfreq(nf, 1/FS), device=model.device)
        ev, vec = torch.linalg.eigh(self.cov)
        self.rank_min = int((ev > ev[:, -1:]*1e-10).sum(-1).min().cpu())
        model.counts['covariance_decompositions'] += nf
        if method == 'MUSIC':
            en = vec[:, :, :4-count]
            self.matrix = en@en.mH
        else:
            scale = self.cov.diagonal(dim1=1, dim2=2).real.mean(-1)
            loaded = self.cov+loading*scale[:, None, None]*torch.eye(4, device=model.device)
            self.matrix = torch.linalg.inv(loaded)
            model.counts['covariance_decompositions'] += nf

    def evaluate(self, points, chunk=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(points), chunk):
                self.m.check()
                a = self.m.response(points[i:i+chunk], self.freq)
                denom = (a.conj()*(self.matrix@a)).sum(1).real.clamp_min(1e-15)
                # MUSIC: reciprocal pooled noise-subspace energy, equivalent to
                # maximizing pooled signal-subspace energy for unit-norm response.
                v = 1/denom.mean(0) if self.method == 'MUSIC' else (1/denom).mean(0)
                out.extend(v.cpu().tolist())
                self.m.counts['candidate_points'] += a.shape[-1]
        return np.asarray(out)


def sbl_posterior(a, x, gamma, noise):
    """Complex Gaussian posterior via LxL Woodbury; no QxQ matrix is built."""
    cov = (a*gamma[None, None])@a.mH+noise*torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
    chol = torch.linalg.cholesky(cov)
    invx = torch.cholesky_solve(x[..., None], chol).squeeze(-1)
    inva = torch.cholesky_solve(a, chol)
    mu = gamma[None]*(a.conj()*invx[:, :, None]).sum(1)
    diagonal = (gamma[None]-gamma[None]**2*(a.conj()*inva).sum(1).real).clamp_min(0)
    evidence = 2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(-1)+(x.conj()*invx).sum(-1).real
    return mu, diagonal, cov, evidence
