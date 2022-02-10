#!/usr/bin/python3

# TODO: use splrep (splines) to interpolate equation of state instead of linear interpolation?

from constants import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

Nc = 3
Nf = 2
mσ = 700 # TODO: important! mσ=550 means not always a minimum, but mσ=800 does!
mπ = 138
mq0 = 300
fπ = 93
me = 0

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

def mq(σ): return g * σ
def pf(μ, m): return np.real(np.sqrt(μ**2 - m**2 + 0j)) # TODO: correct?
#def pf(μ, m): return np.nan if μ**2-m**2 < 0 else np.sqrt(np.maximum(μ**2 - m**2, 0)) # TODO: correct?
    # assert μ**2-m**2 >= -1e-2, f"very bad {μ**2-m**2}"
    #return np.sqrt(np.maximum(μ**2 - m**2, 0))
def xf(μ, m): return pf(μ, m) / np.abs(m)
def n(μ, m): return pf(μ, m)**3 / (3*π**2)

def μelim(σ, μu, μdmax=1e3, verbose=True, ftol=1e-5):
    def f(μd):
        μe = μd - μu # chemical equilibrium
        ret = 2/3 * n(μu,mq(σ)) - 1/3 * n(μd,mq(σ)) - 1 * n(μe,me) # charge neutrality
        return -ret # negate f(μd), so it has a min instead of max, enabling use of minimize_scalar

    # first find extremum (practical to use as one endpoint in bisection method later)
    # then find the root to its right
    # TODO: sometimes there is a root for μd < mq(σ). is this root relevant?
    μd = scipy.optimize.minimize_scalar(f, bounds=(0, μdmax), method="bounded").x # use as endpoint on next line
    assert f(μd) < ftol, f"f(μd = {μd}) = {f(μd)} >= ftol = {ftol}, no solution to charge neutrality"
    if f(μd) < 0: # if function has a root, find it (otherwise proceed with approximate root)
        μd = scipy.optimize.root_scalar(f, bracket=(μd, μdmax), method="bisect").root # final value
    μe = μd - μu
    return μu, μd, μe

def ω(σ, μu, μd=None, μe=None, verbose=False):
    if type(σ) == np.ndarray:
        return np.array([ω(σ, μu, μd=μd, μe=μe, verbose=verbose) for σ in σ])
    if verbose:
        print(f"ω(σ={σ}, μu={μu}); mq = {mq(σ)}")

    if μd is None or μe is None:
        if verbose:
            print("eliminating", μd, μe)
        μu, μd, μe = μelim(σ, μu)

    ω0 = -1/2*m**2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq(σ)**4/(16*π**2)*(3/2+np.log(Λ**2/mq(σ)**2))
    ωe = -μe**4 / (12*π**2)
    xu = xf(μu, mq(σ))
    xd = xf(μd, mq(σ))
    ωu = -Nc/(24*π**2) * mq(σ)**4 * ((2*xu**3-3*xu)*np.sqrt(xu**2+1) + 3*np.arcsinh(xu))
    ωd = -Nc/(24*π**2) * mq(σ)**4 * ((2*xd**3-3*xd)*np.sqrt(xd**2+1) + 3*np.arcsinh(xd))
    return ω0 + ωe + ωu + ωd

def minσ(μu, μd=None, μe=None, σmax=1e2):
    if μd is None:
        μd = [None] * len(μu)
    if μe is None:
        μe = [None] * len(μu)
    #return np.array([scipy.optimize.minimize_scalar(ω, bounds=(0, μu[j]/g), method="bounded", args=(μu[j], μd[j], μe[j])).x for j in range(0, len(μu))])
    return np.array([scipy.optimize.minimize_scalar(ω, bounds=(0, σmax), method="bounded", args=(μu[j], μd[j], μe[j])).x for j in range(0, len(μu))])

"""
# grand potential with μu=μd, μe=0
μu = np.linspace(0, 400, 100)[1:]
σ = np.linspace(-200, +200, 100)
σ0 = minσ(μu, μd=μu, μe=0*μu)
fig, ax = plt.subplots()
for j in range(0, len(μu)):
    ax.plot(σ, ω(σ, μu[j], μd=μu[j], μe=0) / fπ**4)
    ax.scatter(σ0[j], ω(σ0[j], μu[j], μd=μu[j], μe=0) / fπ**4)
plt.show()
"""

# grand potential with charge neutrality and chemical equilibrium
μu = np.linspace(255, 400, 50)
σ = np.linspace(-150, +150, 100)
σ0 = minσ(μu)
ω0 = np.array([ω(σ0[i], μu[i]) for i in range(0, len(μu))])
fig, ax = plt.subplots()
for j in range(0, len(μu)):
    ax.plot(σ, ω(σ, μu[j]) / fπ**4, "-k")
ax.plot(σ0, ω0 / fπ**4, "-ro")
plt.show()

# equation of state with charge neutrality and chemical equilibrium
μu = np.linspace(255, 400, 50) # TODO: phase transition 255-256
σ0 = minσ(μu)
μ = np.array([μelim(σ0[j], μu[j]) for j in range(0, len(μu))])
μu, μd, μe = μ[:,0], μ[:,1], μ[:,2]
nu, nd, ne = n(μu,mq(σ0)), n(μd,mq(σ0)), n(μe,mq(σ0))
ω0 = np.array([ω(σ0[i], μu[i]) for i in range(0, len(μu))])
P0 = -ω0
ϵ0 = μe*ne + μu*nu + μd*nd - P0
# TODO: how to subtract zero-pressure?
plt.plot(P0 / fπ**4, ϵ0 / fπ**4)
plt.xlabel(r"$P$")
plt.ylabel(r"$\epsilon$")
plt.show()

# which value of μ does σ=fπ correspond to?
plt.plot(μu, σ0)
plt.axhline(fπ, color="black", linestyle="dashed", label=f"$f_\\pi = {fπ}$")
plt.xlabel(r"$\mu_u$")
plt.ylabel(r"$\sigma$")
plt.legend()
plt.show()

# before subtraction
plt.plot(σ0, P0)
plt.axhline(0, color="black", linestyle="dashed")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$P$")
plt.show()

P0 = P0 - P0[0] # subtract vacuum pressure (at σ=fπ=93, see previous plot?)
ϵ0 = μe*ne + μu*nu + μd*nd - P0 # TODO: also subtract from energy density??

plt.plot(σ0, P0)
plt.axhline(0, color="black", linestyle="dashed")
plt.xlabel(r"$\sigma$")
plt.ylabel(r"$P$")
plt.show()

plt.plot(P0, ϵ0)
plt.xlabel(r"$P$")
plt.ylabel(r"$\epsilon$")
plt.show()
