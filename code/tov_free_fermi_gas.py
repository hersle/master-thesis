#!/usr/bin/python3

from constants import *
from tov_solve import *
from stability import *
import utils
import numpy as np
import scipy.optimize

# Non-relativistic limit
def ϵNR(P):
    if P <= 0:
        return 0
    prefactor = (5**3*4**2 / (3**2*π**2*b**2) * mn**8*c**6*r0**6 / (m0**2*ħ**6))**(1/5)
    return prefactor * P**(3/5)

# Arbitrary Fermi momentum
def ϵGR(P):
    if P <= 0:
        return 0
    prefactor = mn**4*c**3*r0**3 / (6*π*b*m0*ħ**3)
    def f(x):
        Px = prefactor * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x))
        #asinhx4 = 4*np.arcsinh(x)
        #Px = 24*prefactor * (1/3*x**3*np.sqrt(1+x**2) - (np.sinh(asinhx4)-asinhx4)/32)
        return Px - P
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e5))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵx = 3*prefactor * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    #asinhx4 = 4*np.arcsinh(x)
    #ϵx = 4*mn**4*c**3*r0**3 / (π*b*m0*ħ**3) * (np.sinh(asinhx4)-asinhx4)/32
    return ϵx

# numerical instability for P2 > 1e7
#massradiusplot(ϵNR, (1e-6, 1e7), tolD=0.05, tolP=1e-5, maxdr=1e-3, stability=True, visual=True, outfile="data/nr2.dat")
#massradiusplot(ϵGR, (1e-6, 1e7), tolD=0.05, tolP=1e-5, maxdr=1e-3, stability=True, visual=True, outfile="data/gr2.dat")

"""
for P0 in np.geomspace(1e-6, 1e7, 10):
    r, m, P, α, ϵ = soltov(ϵNR, P0)
    ω2, u = eigenmode(r, m, P, α, ϵ, 0, progress=True)
"""

r, m, P, α, ϵ = soltov(ϵGR, 1e3)
ω2s, us = eigenmode(r, m, P, α, ϵ, [0], plot=True, outfileshoot="data/shoot.dat")# , outfile="data/nmodes.dat")
exit()

#P0 = 1e-1
P0 = 5e2
r, m, P, α, ϵ = soltov(ϵGR, P0)
ns = range(0, 5)
ω2s, us = eigenmode(r, m, P, α, ϵ, ns, plot=False, cut=True, normalize=False, outfile="data/nmodes.dat")
ω2s, us = eigenmode(r, m, P, α, ϵ, ns, plot=False, cut=True, normalize=True, outfile="data/nmodes_norm.dat")
for ω2, u, n in zip(ω2s, us, ns):
    plt.plot(r, u, label=f"n = {n}, ω2 = {ω2}")
plt.legend()
plt.show()
