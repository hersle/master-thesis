#!/usr/bin/python3

# TODO: use splrep (splines) to interpolate equation of state instead of linear interpolation?

from constants import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

Nc = 3
Nf = 2
mσ = 550
mπ = 138
mq0 = 300
fπ = 93
me = 0

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

def mq(σ): return g*(σ)
def pf(μ, m): return np.sqrt(μ**2 - m**2)
def xf(μ, m): return pf(μ, m) / m
def n(μ, m): return pf(μ, m)**3 / (3*π**2)

def μelim(σ, μu, μdmax=1e3, verbose=False):
    def f(μd):
        μe = μd - μu # chemical equilibrium
        ret = 2/3 * n(μu,mq(σ)) - 1/3 * n(μd,mq(σ)) - 1 * n(μe,me) # charge neutrality
        return -ret # negate f(μd), so it has a min instead of max, enabling use of minimize_scalar

    # first find extremum (practical to use as one endpoint in bisection method later)
    # then find the root to its right
    # TODO: sometimes there is a root for μd < mq(σ). is this root relevant?
    μd = scipy.optimize.minimize_scalar(f, bounds=(mq(σ), μdmax), method="bounded").x # use as endpoint on next line
    μd = scipy.optimize.root_scalar(f, bracket=(μd, μdmax), method="bisect").root # final value
    μe = μd - μu
    return μu, μd, μe

def ω(σ, μu, verbose=True):
    if type(σ) == np.ndarray:
        return np.array([ω(σ, μu, verbose=verbose) for σ in σ])
    if verbose:
        print(f"ω(σ={σ}, μu={μu}); mq = {mq(σ)}")

    μu, μd, μe = μelim(σ, μu)

    ω0 = -1/2*m**2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq(σ)**4/(16*π**2)*(3/2+np.log(Λ**2/mq(σ)**2))
    ωe = -μe**4 / (12*π**2)
    ωu = -Nc/(24*π**2) * ((2*μu**2-5*mq(σ)**2)*μu*np.sqrt(μu**2-mq(σ)**2) + 3*mq(σ)**4*np.arcsinh(np.sqrt(μu**2/mq(σ)**2-1)))
    return ω0 + ωe + ωu

def minσ(μu):
    return np.array([scipy.optimize.minimize_scalar(ω, bounds=(0, μu[0]/g), method="bounded", args=(μ)).x for μ in μu])

μu = np.linspace(280, 400, 300)
σ0 = minσ(μu)
μ = np.array([μelim(σ0, μu) for σ0, μu in zip(σ0, μu)])
μu0, μd0, μe0 = μ[:,0], μ[:,1], μ[:,2]

"""
plt.plot(μu, σ0)
plt.plot(μu, μu/g) # should always choose σ0 below this line!
plt.show()
"""

# TODO: plot full landscape and the minimum line
"""
σ = np.linspace(0, μ[0]/g, 100)[1:-1]
#ωs = np.array([[ω(σ,μ) for μ in μ] for σ in σ])
fig, ax = plt.subplots()
for j in range(0, len(μu)):
    ax.plot(σ, ω(σ, μu[j]))
    ax.scatter(σ0[j], ω(σ0[j], μu[j]))
plt.show()
"""

ω0 = np.array([ω(σ0[i], μu0[i]) for i in range(0, len(μu0))])
print(ω0.shape)
P0 = -ω0
ϵ0 = μe0*n(μe0,me) + μu0*n(μu0,mq(σ0)) + μd0*n(μd0,mq(σ0)) - P0

plt.plot(n(μu,mq(σ0)), P0)
plt.plot(n(μu,mq(σ0)), ϵ0)
plt.show()

# before subtraction of zero-pressure
"""
plt.plot(P0, ϵ0, "-k")
plt.show()
"""

# TODO: is this equivalent to subtracting the P(ne=nu=nd=0) (this sounds like a stronger definition of vacuum than ϵ=0)
# subtract (interpolated) value P(ϵ=0) so that the new pressure has P(ϵ=0)=0
P0 -= np.interp(0, ϵ0, P0)

# TODO: remove negative values of P (and corresponding ones of ϵ), then insert (0,0) as first point
P0 = np.insert(P0[ϵ0 >= 0], 0, 0)
ϵ0 = np.insert(ϵ0[ϵ0 >= 0], 0, 0)

# after subtraction of zero-pressure
plt.plot(P0 / fπ**4, ϵ0 / fπ**4, "-k")
plt.xlabel(r"$P$")
plt.ylabel(r"$\epsilon$")
plt.show()
