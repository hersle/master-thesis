#!/usr/bin/python3

from constants import *
from tov import *
from stability import *
from utils import *
import numpy as np
import scipy.optimize

def ϵUR(P): # ultra-relativistic equation of state
    return 3 * P

def ϵNR(P): # non-relativistic equation of state
    if P <= 0: return 0
    prefactor = (5**3*4**2 / (3**2*π**2*b**2) * mn**8*c**6*r0**6 / (m0**2*ħ**6))**(1/5)
    return prefactor * P**(3/5)

def ϵGR(P): # general equation of state
    if P <= 0: return 0
    prefactor = mn**4*c**3*r0**3 / (6*π*b*m0*ħ**3)
    def f(x):
        Px = prefactor * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x))
        return Px - P
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e5))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵx = 3*prefactor * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    return ϵx

# Write equations of state
P = np.linspace(0, 20, 500)
ϵs = []
for ϵ in (ϵUR, ϵNR, ϵGR):
    ϵs.append([ϵ(P) for P in P])
writecols([P, *ϵs], ["P", "epsUR", "epsNR", "epsGR"], "data/eos.dat")

opts = { "tolD": 0.05, "tolP": 1e-5, "maxdr": 1e-3, "visual": True }
massradiusplot(
    ϵNR, (1e-6, 1e0), **opts, nmodes=0, newtonian=True, outfile="data/nrnewt.dat"
)
massradiusplot(
    ϵGR, (1e-6, 1e0), **opts, nmodes=0, newtonian=True, outfile="data/grnewt.dat"
)
massradiusplot(
    ϵNR, (1e-6, 1e7), **opts, nmodes=0,  newtonian=False, outfile="data/nr.dat"
)
massradiusplot(
    ϵGR, (1e-6, 1e7), **opts, nmodes=6,  newtonian=False, outfile="data/gr.dat"
)

P0s = list(np.geomspace(1e-6, 1e7, 14))
xs, ps = [], []
for P0 in P0s:
    r, m, P, α, ϵ = soltov(ϵGR, P0)
    xs.append(list(r / r[-1]))
    ps.append(list(P / P0))
P0head = ["P0"]
xheads = [f"x{i}" for i in range(0, len(P0s))]
pheads = [f"p{i}" for i in range(0, len(P0s))]
writecols([P0s] + xs + ps, P0head + xheads + pheads, "data/pressures.dat")

r, m, P, α, ϵ = soltov(ϵGR, 1e3)
ω2s, us = eigenmode(r, m, P, α, ϵ, [0], plot=True, outfileshoot="data/shoot.dat")

P0 = 3e2
r, m, P, α, ϵ = soltov(ϵGR, P0)
ns = range(0, 12)
ω2s, us = eigenmode(
    r, m, P, α, ϵ, ns, cut=True, normalize=False, outfile="data/nmodes.dat"
)
ω2s, us = eigenmode(
    r, m, P, α, ϵ, ns, cut=True, normalize=True, outfile="data/nmodes_norm.dat"
)
