import torch
import numpy as np
import timeit
import scipy.stats as sns

from utils import get_rand_SU3
from diag import _np_diagonalize, _torch_diagonalize, _torch_diagonalize_su3


def test(bs=10):
    U = get_rand_SU3(bs)

    def diag_undiag(f):
        d, P = f(U)
        assert torch.allclose(P @ (d[..., np.newaxis,:] * torch.eye(3)) @ P.adjoint(), U)
        print("Done:", f)

    for f in [_np_diagonalize, _torch_diagonalize, _torch_diagonalize_su3]:
        diag_undiag(f)


def benchmark(bs):
    U = get_rand_SU3(bs)
    globals().update({'U': U})
    for f in ['_np_diagonalize', '_torch_diagonalize', '_torch_diagonalize_su3']:
        times = np.array(
            timeit.repeat(
                f"torch.cuda.synchronize(); d, P = {f}(U) ;torch.cuda.synchronize();",
                globals=globals(), number=10, repeat=10)
            )
        print(f"{f} time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')


if __name__ == '__main__':
    assert torch.cuda.is_available(), \
        "Script tests eigenproblem solver on GPU, CUDA must be available."
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    # test()
    print("small lattice")
    benchmark(4**4*8)
    print("large lattice")
    benchmark(8**4*8)

