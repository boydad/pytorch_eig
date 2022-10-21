import torch
import numpy as np
import timeit
import scipy.stats as sns

import nvtx

from utils import get_rand_SU3, nvtx_decorator
from diag import _np_diagonalize, _torch_diagonalize, _torch_diagonalize_su3


def benchmark(diag_fun, *, batch_size=8,
              warmup=5, num_runs=10, repeat=10, preprocess=lambda x: x):
    U = get_rand_SU3(batch_size)
    U = preprocess(U)

    # Warmup
    for _ in range(warmup):
        diag_fun(U)

    results = np.array(
        timeit.repeat(
            f"d, P = diag_fun(U)",
            globals=locals(), number=num_runs, repeat=10)
        ) / num_runs
    return results


def print_benchmark(results, *, prefix):
    low = 1000 * np.min(results)
    high = 1000 * np.max(results)

    print(f"{prefix}: {low:.2f}-{high:.2f}ms")


if __name__ == '__main__':
    with torch.autograd.profiler.emit_nvtx():

        print('Lattice 4**4 x 8')
        bs = 4**4*8

        prep = lambda U: U.real + U.real.transpose(-1,-2)
        # prep = lambda U: U

        f = nvtx_decorator(_np_diagonalize, 'np')
        print_benchmark(benchmark(f, batch_size=bs, preprocess=prep), prefix=f)

        f = nvtx_decorator(_torch_diagonalize, 'torch')
        print_benchmark(benchmark(f, batch_size=bs, preprocess=prep), prefix=f)

        f = nvtx_decorator(_torch_diagonalize_su3, 'custom')
        print_benchmark(benchmark(f, batch_size=bs, preprocess=prep), prefix=f)

        f = nvtx_decorator(torch.jit.script(_torch_diagonalize_su3), 'jit')
        print_benchmark(benchmark(f, batch_size=bs, preprocess=prep), prefix=f)

        # print('Lattice 8**4 x 8')
        # bs = 8**4 * 8
        # for f in [_np_diagonalize,
        #           _torch_diagonalize,
        #           _torch_diagonalize_su3,
        #           torch.jit.script(_torch_diagonalize_su3)]:
        #     print_benchmark(benchmark(f, batch_size=bs), prefix=f)
