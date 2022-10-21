import torch
import numpy as np

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


if __name__ == '__main__':
    assert torch.cuda.is_available(), \
        "Script tests eigenproblem solver on GPU, CUDA must be available."
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    test()
    