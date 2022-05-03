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
        assert torch.allclose(P @ (d[..., np.newaxis,:] * torch.eye(3, device = U.device)) @ P.adjoint(), U)
        print("Done:", f)

    for f in [_np_diagonalize, _torch_diagonalize, _torch_diagonalize_su3]:
        diag_undiag(f)


def benchmark(bs):
    U = get_rand_SU3(bs)
    globals().update({'U': U})
    for f in ['_np_diagonalize', '_torch_diagonalize', '_torch_diagonalize_su3']:
        times = np.array(
            timeit.repeat(
                f"torch.cuda.synchronize(); d, P = {f}(U); torch.cuda.synchronize();",
                globals=globals(), number=10, repeat=10)
            )
        print(f"{f} time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')

def benchmark_fwd_bwd(bs):
    U = get_rand_SU3(bs)
    U.requires_grad_(True)
    globals().update({'U': U})
    for f in ['_torch_diagonalize', '_torch_diagonalize_su3']:
        times = np.array(
            timeit.repeat(
                f"torch.cuda.synchronize(); d, P = {f}(U); loss = torch.sum(d); loss.backward(); torch.cuda.synchronize();",
                globals=globals(), number=10, repeat=10)
        )
        print(f"{f} time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')
        

if __name__ == '__main__':
    assert torch.cuda.is_available(), \
        "Script tests eigenproblem solver on GPU, CUDA must be available."
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    #torch.backends.cudnn.benchmark = True

    # nvfuser
    #torch._C._jit_set_profiling_executor(True)
    #torch._C._jit_set_profiling_mode(True)
    #torch._C._jit_override_can_fuse_on_cpu(False)
    #torch._C._jit_override_can_fuse_on_gpu(False)
    #torch._C._jit_set_texpr_fuser_enabled(False)
    #torch._C._jit_set_nvfuser_enabled(True)
    #torch._C._debug_set_autodiff_subgraph_inlining(False)

    test()
    print("small lattice")
    benchmark_fwd_bwd(4**4*8)
    print("large lattice")
    benchmark_fwd_bwd(8**4*8)

