#!/usr/bin/python3

# TODO: use splrep (splines) to interpolate equation of state instead of linear interpolation?

from constants import *
import utils

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

Nc = 3
Nf = 2
mσ = 800 # TODO: important! mσ=550 means not always a minimum, but mσ=800 does!
mπ = 138
mq0 = 300
fπ = 93
me = 0

# TODO: discuss three parameter choices: (neutral=False,μside=any), (neutral=True,μside=left), (neutral=True,μside=right)
neutral = True
μside = "left" # TODO: is left or right charge neutrality solution correct? left/right gives few/many up and many/few electrons

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

# results:
# 1. mπ=0, neutral=False, μside=arbitrary: discontinuous phase transition around μ = 320 MeV (chiral limit)

def mq(σ): return g * σ
def pf(μ, m): return np.real(np.sqrt(μ**2 - m**2 + 0j)) # TODO: correct?
def xf(μ, m): return pf(μ, m) / np.abs(m)
def n(μ, m): return pf(μ, m)**3 / (3*π**2)
def q(σ, μ):
    qu, qd, qe = +2/3, -1/3, -1
    μu, μd, μe = μ[0], μ[1], μ[2]
    mu, md = mq(σ), mq(σ) # me is global
    return qu * n(μu,mu) - qd * n(μd,md) - qe * n(μe,me) # charge neutrality

def μelim(σ, μu, neutral, side, μdmax=1e3, verbose=False, ftol=1e-5):
    assert np.shape(σ) == np.shape(μu), "σ and μu has different shapes"
    if np.ndim(σ) >= 1:
        return np.array([μelim(σ[i], μu[i], μdmax=μdmax, verbose=verbose, ftol=ftol, neutral=neutral, side=side) for i in range(0, len(σ))])
        
    if not neutral:
        μd = μu
        μe = 0
    else:
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
            if side == "left":
                μd = scipy.optimize.root_scalar(f, bracket=(0, μd), method="bisect").root # final value (left solution)
            elif side == "right":
                μd = scipy.optimize.root_scalar(f, bracket=(μd, μdmax), method="bisect").root # final value (right solution)
            else:
                assert False, "side must be left or right"
        μe = μd - μu

    return np.array([μu, μd, μe])

def ωf(σ, μ, verbose=False):
    #assert np.shape(σ) == np.shape(μu) == np.shape(μd) == np.shape(μe), "σ and μ have different shapes"
    if np.ndim(σ) >= 1:
        return np.array([ωf(σ[i], μ[i], verbose=verbose) for i in range(0, len(σ))])

    if verbose:
        print(f"ω(σ={σ}, μ={μ}); mq = {mq(σ)}")

    μu, μd, μe = μ[0], μ[1], μ[2]
    ω0 = -1/2*m**2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq(σ)**4/(16*π**2)*(3/2+np.log(Λ**2/mq(σ)**2))
    ωe = -μe**4 / (12*π**2)
    xu = xf(μu, mq(σ))
    xd = xf(μd, mq(σ))
    ωu = -Nc/(24*π**2) * mq(σ)**4 * ((2*xu**3-3*xu)*np.sqrt(xu**2+1) + 3*np.arcsinh(xu))
    ωd = -Nc/(24*π**2) * mq(σ)**4 * ((2*xd**3-3*xd)*np.sqrt(xd**2+1) + 3*np.arcsinh(xd))
    return ω0 + ωe + ωu + ωd

def σμω(σ, μu, neutral, μside, verbose=False):
    μ = μelim(σ, μu, neutral, μside) # [nσ, nμu, 3]
    ω = ωf(σ, μ)
    μu, μd, μe = μ[...,0], μ[...,1], μ[...,2]
    return σ, μu, μd, μe, ω

def σμω0(μu, neutral, μside, σmax=200):
    if np.ndim(μu) >= 1:
        return np.array([σμω0(μu, neutral, μside, σmax=σmax) for μu in μu])

    def ωf2(σ):
        μus = np.full(np.shape(σ), μu)
        _, _, _, _, ω = σμω(σ, μus, neutral, μside)
        return ω
    σ0 = scipy.optimize.minimize_scalar(ωf2, bounds=(0, σmax), method="bounded").x
    μ0 = μelim(σ0, μu, neutral, μside)
    μu0, μd0, μe0 = μ0[...,0], μ0[...,1], μ0[...,2]
    ω0 = ωf(σ0, μ0)
    return σ0, μu0, μd0, μe0, ω0

# grand potential
σ = np.linspace(-200, +200, 50)
μu = np.linspace(0, 600, 60)
μuμu, σσ = np.meshgrid(μu, σ)
if True:
    σ, μu, μd, μe, ω = σμω(σσ, μuμu, neutral, μside)
    ωc = ω.reshape(-1)
    μuc = μu.reshape(-1)
    σc = σ.reshape(-1)
    utils.writecols([σc, μuc, ωc / fπ**4], ["sigma", "muu", "omega"], "data/lsmpot_neutral_left.dat", skipevery=np.shape(ω)[1])
    σ, μu, μd, μe = σ[:,0], μu[0,:], μd[0,:], μe[0,:]
    plt.plot(σ, ω / fπ**4, "-k")
ret = σμω0(μu, neutral, μside)
σ0, μu0, μd0, μe0, ω0 = ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4]
pti = np.argmax(np.abs(σ0[1:]-σ0[:-1])) + 1 # pti is now first point after phase transition
if np.abs(σ0[pti]-σ0[pti-1]) < 50:
    print("no pt")
    pti = 0 # no phase transition
print(f"phase transition between μu = {μu[pti-1]}, {μu[pti]}")
cols = np.array([σ0, μu0, μd0, μe0, ω0 / fπ**4])
print(cols.shape)
cols = np.concatenate([cols[:,:pti], np.full((len(cols),1), np.nan), cols[:,pti:]], axis=1) # add nan at phase transition
print(cols.shape)
utils.writecols(cols, ["sigma", "muu", "mud", "mue", "omega"], "data/lsmpot_neutral_left_min.dat")
plt.plot(σ0, ω0 / fπ**4, "-ro")
plt.show()

# plot minimum line
plt.plot(σ0, μu0)
plt.show()

# verify charge neutrality
"""
if neutral:
    plt.plot(μu, 2/3*n(μu0,mq(σ0))-1/3*n(μd0,mq(σ0))-1*n(μe0,me))
    plt.show()
"""

# show densities
# TODO: is it correct that the electron density so extremely low?
nu0, nd0, ne0 = n(μu0, mq(σ0)), n(μd0, mq(σ0)), n(μe0, me)
plt.plot(μu0, nu0, "-r.")
plt.plot(μd0, nd0, "-g.")
plt.plot(μe0, ne0, "-b.")
plt.show()

# equation of state
P0 = -ω0
P0 = P0 - P0[0] # subtract vacuum contribution P(fπ) TODO: correct?
#print(P0)
#plt.plot(μu0[pti:], P0[pti:])
ϵ0 = μe0*ne0 + μu0*nu0 + μd0*nd0 - P0
plt.plot(P0[pti:], ϵ0[pti:])
plt.show()
