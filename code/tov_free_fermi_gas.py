#!/usr/bin/python3

from tov_solve import massradiusplot, r0, m0
import utils
import numpy as np
import scipy.optimize

from constants import *
m = 1.67492749804e-27 # kg

# Non-relativistic limit
def ϵNR(P):
    factorNR = 5**3*4**2 / (3**4*π**2) * m**8*c**6*r0**6 / (m0**2*ħ**6)
    return (factorNR * np.abs(P)**3)**(1/5)

massradiusplot(ϵNR, (1e-6, 1e21), tolD=0.05, tolP=1e-5, maxdr=2e-3, outfile="data/nr.dat")

# Arbitrary Fermi momentum
def ϵGR(P):
    P = np.abs(P) # avoid (small) negative pressure TODO: implications of this?
    factorGR = m**4*c**3*r0**3 / (18*π*m0*ħ**3)
    f = lambda x, P: factorGR * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x)) - P
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e18), args=(P))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵ = 3*factorGR * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    return ϵ

massradiusplot(ϵGR, (1e-6, 1e17), tolD=0.1, tolP=1e-5, maxdr=2e-3, outfile="data/gr.dat")
