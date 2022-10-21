"""Autograd functions to diagonalize Lie group elements"""
import numpy as np
import math
import torch


def _np_diagonalize(U):
    d, P = np.linalg.eig(U.detach().cpu().numpy())
    d = torch.from_numpy(d).to(device=U.device, dtype=U.dtype)
    P = torch.from_numpy(P).to(device=U.device, dtype=U.dtype)
    return d, P

def _torch_diagonalize(U):
    d, P = torch.linalg.eig(U)
    return d, P

def _torch_diagonalize_su3(U):
    """Pytorch implementation of analytic solution of SU(3) diagonalization"""
    shape = U.shape
    U = U.reshape(-1,9)
    a,b,c,p,q,r,x,y,z = U[:,0], U[:,1], U[:,2], U[:,3], U[:,4], U[:,5], U[:,6], U[:,7], U[:,8]

    sqrt3 = math.sqrt(3)
    
    term1 = (a + q + z)/3.
    term2 = -a**2 - 3*b*p + a*q - q**2 - 3*c*x - 3*r*y + a*z + q*z - z**2
    term3 = 2*a**3 + 9*a*b*p - 3*a**2*q + 9*b*p*q - 3*a*q**2 + 2*q**3 +\
        9*a*c*x - 18*c*q*x + 27*b*r*x + 27*c*p*y - 18*a*r*y + 9*q*r*y -\
        3*a**2*z - 18*b*p*z + 12*a*q*z - 3*q**2*z + 9*c*x*z + 9*r*y*z -\
        3*a*z**2 - 3*q*z**2 + 2*z**3
    term4 = (term3 + torch.sqrt(4*(term2)**3 + (term3)**2))**(1/3)
    term5 = (2**(1/3)*(term2))/(3.*term4)
    term6 = (term2*(1 + sqrt3*1j))/(term4*3.*2**(2/3))
    term7 = (term4*(1 - sqrt3*1j))/(6.*2**(1/3))
    term8 = (term2*(1 - sqrt3*1j))/(term4*3.*2**(2/3))
    term9 = (term4*(1 + sqrt3*1j))/(6.*2**(1/3))
    term10 = -(r*x) + p*z
    term11 = -(q*x) + p*y
    term12 = r*x - p*z

    eigval1 = term1 - term5 + term4/(3.*2**(1/3))
    eigval2 = term1 + term6 - term7
    eigval3 = term1 + term8 - term9
    eigvals = torch.stack([eigval1, eigval2, eigval3], dim=1)

    eigvec1 = torch.stack([
        (-z + eigval1)/x + (y*(term10 - p*(eigval1)))/(x*(term11 + x*(eigval1))),
        (term12 + p*(eigval1))/(term11 + x*(eigval1)),
        torch.ones_like(a)], dim=1).unsqueeze(2)
    eigvec2 = torch.stack([
        (-z + eigval2)/x + (y*(term10 - p*(eigval2)))/(x*(term11 + x*(eigval2))),
        (term12 + p*(eigval2))/(term11 + x*(eigval2)),
        torch.ones_like(a)], dim=1).unsqueeze(2)
    eigvec3 = torch.stack([
        (-z + eigval3)/x + (y*(term10 - p*(eigval3)))/(x*(term11 + x*(eigval3))),
        (term12 + p*(eigval3))/(term11 + x*(eigval3)),
        torch.ones_like(a)], dim=1).unsqueeze(2)
    eigvec1, eigvec2, eigvec3 = (
        eigvec1 / torch.linalg.norm(eigvec1, dim=1).unsqueeze(1),
        eigvec2 / torch.linalg.norm(eigvec2, dim=1).unsqueeze(1),
        eigvec3 / torch.linalg.norm(eigvec3, dim=1).unsqueeze(1))
    eigvecs = torch.cat([eigvec1, eigvec2, eigvec3], dim=2)

    return eigvals.reshape(shape[:-1]), eigvecs.reshape(shape)
