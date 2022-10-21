import torch
import numpy as np
from scipy.stats import unitary_group


def get_rand_SU3(batch_size, device="cuda"):
    U = torch.from_numpy(
            np.array([
                unitary_group.rvs(3).astype(np.cdouble)
                for _ in range(batch_size)]
            )
        ).to(device)
    phases = torch.det(U)**(1/3)
    U = U * phases[:, np.newaxis, np.newaxis].conj()  # -> det U = 1
    # randomize over center elements
    centers = torch.exp(
        1j * 2 * np.pi / 3 * 
        torch.randint(0, 3, (batch_size,), device=device)
    )
    U = U * centers[..., np.newaxis, np.newaxis]
    return U


def nvtx_decorator(f, name):
    def decorated(*args, **kwars):
        torch.cuda.nvtx.range_push(name)
        tmp = f(*args, **kwars)
        torch.cuda.nvtx.range_pop()
        return tmp
    return decorated
