"""Exact posterior-kernel equivalence and speed, separate from localization evidence."""
import time
import numpy as np
import torch
from ..physics import sbl_posterior
from .runtime import preflight, fresh, write


def posterior_fast(a, x, gamma, noise):
    cov = (a*gamma[None, None])@a.mH+noise*torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
    chol = torch.linalg.cholesky(cov)
    inv = torch.cholesky_inverse(chol)
    invx = (inv@x[..., None]).squeeze(-1)
    inva = inv@a
    mu = gamma[None]*(a.conj()*invx[:, :, None]).sum(1)
    diagonal = (gamma[None]-gamma[None]**2*(a.conj()*inva).sum(1).real).clamp_min(0)
    evidence = 2*torch.log(chol.diagonal(dim1=1, dim2=2).real).sum(-1)+(x.conj()*invx).sum(-1).real
    return mu, diagonal, cov, evidence


def main():
    out = fresh('kernel_check')
    preflight(out, dict(stage='ENGINEERING_KERNEL_CHECK', not_performance=True))
    torch.manual_seed(93)
    a = torch.randn((256, 4, 10201), dtype=torch.complex128, device='cuda')
    x = torch.randn((256, 4), dtype=torch.complex128, device='cuda')
    gamma = torch.ones(10201, device='cuda', dtype=torch.float64)*.0001
    ref = sbl_posterior(a, x, gamma, .01)
    fast = posterior_fast(a, x, gamma, .01)
    for u,v in zip(ref, fast):
        np.testing.assert_allclose(u.cpu(), v.cpu(), rtol=1e-10, atol=1e-12)
    times = {}
    for name, fn in [('reference',sbl_posterior),('small_inverse',posterior_fast)]:
        for _ in range(2):
            fn(a,x,gamma,.01)
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(10):
            fn(a,x,gamma,.01)
        torch.cuda.synchronize()
        times[name]=(time.perf_counter()-start)/10
    write(out/'report.json', dict(passed=True, times=times, shape=list(a.shape)))
    print(out, times, flush=True)


if __name__ == '__main__':
    main()
