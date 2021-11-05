#!/usr/bin/python3

from constants import *
from tov_solve import massradiusplot, r0, m0
import utils
import numpy as np

def ϵUR(P): return 3*P

m = 1.67492749804e-27 # kg
ratio = m**8*c**6*r0**6 / (m0**2*ħ**6)

def ϵNR(P): return (5**3*4**2 / (3**4*π**2) * ratio * np.abs(P)**3)**(1/5)

# Px(1e7) ≈ 1e28, so need to consider x ∈ [0, 1e7] for P ∈ [0, 1e28]
def Px(x): return m**4*c**3*r0**3 / (18*π*m0*ħ**3) * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x))
def ϵGR(P):
    P = np.abs(P) # TODO: implications?
    def f(x): return Px(x) - P
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e18))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵ = m**4*c**3*r0**3 / (6*π*m0*ħ**3) * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    return ϵ

# rs, ms, Ps = soltov(ϵNR, 2.4)

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-6, 1e10, tolD=0.05, tolP=1e-5, max_step=5e-4, visual=True) # for full curve

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e1, 1e21, tolD=0.04, tolP=1e-3, max_step=2e-4, visual=True) # for spiral only

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-6, 1e21, tolD=0.05, tolP=1e-5, max_step=2e-4, visual=True) # everything?
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/nr.dat")

# TODO: P never 0, cannot integrate to np.inf
#Ps, Ms, Rs = massradiusplot(ϵUR, 1e-0, 1e4, tolD=0.1, tolP=1e-1, max_step=1e-2, visual=True) # everything?
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/ur.dat")

# Everything for GR
#Ps, Ms, Rs = massradiusplot(ϵGR, 1e-6, 1e17, tolD=0.05, tolP=1e-5, max_step=2e-3, visual=True)
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/gr.dat")

# TESTING
Ps, Ms, Rs = massradiusplot(ϵNR, 1e-6, 1e3, tolD=0.5, tolP=1e-5, visual=False) # for full curve
utils.writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/test.dat")


#plt.plot(Rs, Ms, "-ko")
#plt.show()
#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-2, 1e2, tolD=0.5, tolP=1e-2, max_step=1e-3, visual=False) # for spiral
