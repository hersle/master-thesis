#!/usr/bin/python3

from constants import *
from tov_solve import *
import utils
import numpy as np
import scipy.optimize

# Non-relativistic limit
def ϵNR(P):
    if P <= 0:
        return 0
    return (5**3*4**2 / (3**2*π**2*b**2) * m**8*c**6*r0**6 / (m0**2*ħ**6) * np.abs(P)**3)**(1/5)

# Arbitrary Fermi momentum
def ϵGR(P):
    if P <= 0:
        return 0
    f = lambda x: m**4*c**3*r0**3 / (6*π*b*m0*ħ**3) * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x)) - P
    #f = lambda x, P: 4*m**4*c**3*r0**3 / (π*b*m0*ħ**3) * (1/3*x**3*np.sqrt(1+x**2) - (np.sinh(4*np.arcsinh(x))-4*np.arcsinh(x))/32) - P # alternative
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e5))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵ = m**4*c**3*r0**3 / (2*π*b*m0*ħ**3) * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    #ϵ = 4*m**4*c**3*r0**3 / (π*b*m0*ħ**3) * (np.sinh(4*np.arcsinh(x))-4*np.arcsinh(x))/32 # alternative
    return ϵ

massradiusplot(ϵNR, (1e-6, 1e7), tolD=0.05, tolP=1e-5, maxdr=1e-3, outfile="data/nr.dat", visual=True)
massradiusplot(ϵGR, (1e-6, 1e7), tolD=0.05, tolP=1e-5, maxdr=1e-3, outfile="data/gr.dat", visual=True) # numerical instability for P2 > 1e7
