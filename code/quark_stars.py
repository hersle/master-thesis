#!/usr/bin/python3

import constants
π = constants.π
from tov import massradiusplot
import utils

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

Nc = 3
Nf = 2
mσ = 800 # TODO: important! mσ=550 means not always a minimum, but mσ=800 does!
mπ = 138
mq0 = 300
fπ = 93
me = 0

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

# results:
# 1. mπ=0, neutral=False, μside=arbitrary: discontinuous phase transition around μ = 320 MeV (chiral limit)

def mq(σ): return g * σ
def pf(μ, m): return np.real(np.sqrt(μ**2 - m**2 + 0j))
def xf(μ, m): return pf(μ, m) / np.abs(m)
def n(μ, m): return pf(μ, m)**3 / (3*π**2)

def μelim(σ, μu, neutral, side, neglecte, μdmax=1e5, verbose=False, ftol=0):
    assert np.shape(σ) == np.shape(μu), "σ and μu has different shapes"
    if np.ndim(σ) >= 1:
        return np.array([μelim(σ[i], μu[i], neutral, side, neglecte, μdmax=μdmax, verbose=verbose, ftol=ftol) for i in range(0, len(σ))])
        
    if not neutral:
        μd = μu
        μe = 0
    else:
        if neglecte:
            μd = 2**(1/3) * μu # from Two Lectures on Color Superconductivity (Shovkovy), page 1337
            μe = μu/4
            return np.array([μu, μd, μe])
        
        if μu <= mq(σ): # then nu = 0, and ne = nd = 0 is a trivial solution
            μe = 0 # so ne = 0
            μd = μu # from chemical equilibrium (μe = μd - μu)
        else:
            def f(μd):
                μe = μd - μu # chemical equilibrium
                ret = +2/3 * n(μu,mq(σ)) - 1/3 * n(μd,mq(σ)) - 1 * n(μe,me) # charge neutrality
                return -ret # negate f(μd), so it has a min instead of max, enabling use of minimize_scalar

            # first find extremum (practical to use as one endpoint in bisection method later)
            # then find the root to its right
            μd = scipy.optimize.minimize_scalar(f, bounds=(0, μdmax), method="bounded").x # use as endpoint on next line
            μd0 = μd # copy minimum for later reference

            assert f(μd0) < ftol, f"f(μd = {μd0}) = {f(μd0)} >= ftol = {ftol}, no solution to charge neutrality"
            if f(μd0) < 0: # if function has a root, find it (otherwise proceed with approximate root)
                if side == "left":
                    μd = scipy.optimize.root_scalar(f, bracket=(0, μd0), method="bisect").root # final value (left solution)
                elif side == "right":
                    μd = scipy.optimize.root_scalar(f, bracket=(μd0, μdmax), method="bisect").root # final value (right solution)
                else:
                    assert False, "side must be left or right"

            μe = μd - μu

            if False and μu > 300:
                print(f"σ = {σ}, μu = {μu}")
                #print(f"min:  f(μd = {μd0}) = {f(μd0)}")
                #print(f"zero: f(μd = {μd}) = {f(μd)}")
                print(f"u   : nu = {n(μu,mq(σ))}")
                print(f"d   : nd = {n(μd,mq(σ))}")
                print(f"e   : ne = {n(μe,me)}")
                μD = np.linspace(0, μdmax, 2000)
                plt.plot(μD, np.sign(f(μD))*np.log(1+np.abs(f(μD))), "-k.")
                plt.axhline(0, color="black", linestyle="dashed")
                plt.scatter(μd, f(μd))
                plt.xlabel(r"$\mu_d$")
                plt.ylabel(r"$f(\mu_d)$")
                plt.show()

    return np.array([μu, μd, μe])

def ωf(σ, μ, verbose=False):
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
    #ωu = -Nc/(24*π**2) * ((2*μu**2-5*mq(σ)**2)*μu*np.real(np.sqrt(μu**2-mq(σ)**2+0j)) + 3*mq(σ)**4*np.arcsinh(np.real(np.sqrt(μu**2/mq(σ)**2-1+0j))))
    #ωd = -Nc/(24*π**2) * ((2*μd**2-5*mq(σ)**2)*μd*np.real(np.sqrt(μd**2-mq(σ)**2+0j)) + 3*mq(σ)**4*np.arcsinh(np.real(np.sqrt(μd**2/mq(σ)**2-1+0j))))
    return ω0 + ωe + ωu + ωd

def σμω(σ, μu, neutral, μside, neglecte, verbose=False):
    μ = μelim(σ, μu, neutral, μside, neglecte) # [nσ, nμu, 3]
    ω = ωf(σ, μ)
    μu, μd, μe = μ[...,0], μ[...,1], μ[...,2]
    return σ, μu, μd, μe, ω

def σμω0(μu, neutral, μside, neglecte, σmax=200):
    if np.ndim(μu) >= 1:
        return np.array([σμω0(μu, neutral, μside, neglecte, σmax=σmax) for μu in μu])

    def ωf2(σ):
        μus = np.full(np.shape(σ), μu)
        _, _, _, _, ω = σμω(σ, μus, neutral, μside, neglecte)
        return ω
    σ0 = scipy.optimize.minimize_scalar(ωf2, bounds=(0, σmax), method="bounded").x
    μ0 = μelim(σ0, μu, neutral, μside, neglecte)
    μu0, μd0, μe0 = μ0[...,0], μ0[...,1], μ0[...,2]
    ω0 = ωf(σ0, μ0)
    return σ0, μu0, μd0, μe0, ω0

def eos(neutral, μside, neglecte, B=0, name="ϵ", plotfirst=False):
    μu = np.concatenate([np.linspace(0, 250, 50), np.linspace(250, 400, 200), np.linspace(400, 10000, 50)])
    #μu = np.linspace(0, 1000, 500)
    ret = σμω0(μu, neutral, μside, neglecte)
    σ0, μu0, μd0, μe0, ω0 = ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4]
    nu0, nd0, ne0 = n(μu0, mq(σ0)), n(μd0, mq(σ0)), n(μe0, me)
    P0 = -ω0
    P0 = P0 - P0[0]
    P0 -= B # bag constant TODO: which value?
    ϵ0 = μe0*ne0 + μu0*nu0 + μd0*nd0 - P0

    # convert units
    Mev = 1e6 * 1.6e-19
    a, b = -3, -3
    ϵ0 *= Mev**4 * constants.ħ**a * constants.c**b # to SI units
    P0 *= Mev**4 * constants.ħ**a * constants.c**b # to SI units
    ϵ0 /= constants.ϵ0 # to TOV-dimensionless units
    P0 /= constants.ϵ0 # to TOV-dimensionless units

    # cut away 0, linearly interpolate
    #i = np.argmax(ϵ0 > 3e-4)
    #ϵ0 = ϵ0[i:]
    #P0 = P0[i:]
    #ϵx = ϵ0[1]-(ϵ0[2]-ϵ0[1])/(P0[2]-P0[1])*P0[1] # linear interpolate ϵ at P=0
    #P0 = np.concatenate([[0], P0])
    #ϵ0 = np.concatenate([[ϵx], ϵ0])

    ϵ = scipy.interpolate.interp1d(P0, ϵ0)
    ϵ.__name__ = name

    def ϵf(P): return 0 if P <= 0 else ϵ(P)

    if plotfirst:
        plt.plot(P0, ϵ(P0), "-k")
        plt.plot(P0, ϵ0, " r.")
        plt.xlim(0, 0.001)
        plt.ylim(0, 0.001)
        #plt.xticks(np.linspace(0, 1, 11))
        #plt.yticks(np.linspace(0, 1, 11))
        plt.show()

    return ϵf

plotfirst = True
#ϵ1 = eos(False, "", False, name="ϵ", plotfirst=plotfirst)
ϵ2 = eos(True, "right", False, B=0, name="ϵ", plotfirst=plotfirst)
#ϵ3 = eos(True, "right", True, name="ϵ", plotfirst=plotfirst)
#P = np.linspace(0, 1, 5000)
#plt.plot(P, ϵ1(P), "-r.")
#plt.plot(P, ϵ2(P), "-g.")
#plt.plot(P, ϵ3(P), "-b.")
#plt.show()

#opts = { "tolD": 0.10, "tolP": 1e-5, "maxdr": 1e-2, "Psurf": 0, "visual": True }
#massradiusplot(ϵ1, (1e-5, 1e-4), **opts, nmodes=0)

# TODO: test bag model
#B=0
#def ϵ(P): return 3*P + 4*B/constants.ϵ0
#opts = { "tolD": 0.10, "tolP": 1e-5, "maxdr": 1e-3, "Psurf": 0, "visual": True }
#massradiusplot(ϵ, (1e-6, 1e-2), **opts, nmodes=0)


opts = { "tolD": 0.10, "tolP": 1e-5, "maxdr": 1e-3, "Psurf": 0, "visual": True }
massradiusplot(ϵ2, (1e-6, 1e-2), **opts, nmodes=0)

#opts = { "tolD": 0.10, "tolP": 1e-5, "maxdr": 1e-3, "Psurf": 0, "visual": True }
#massradiusplot(ϵ3, (1e2, 1e4), **opts, nmodes=0)

# grand potential
neutral = True
μside = "right" # left or right
neglecte = True
σ = np.linspace(0, +200, 100)[1:]
μu = np.linspace(0, 600, 200)
μuμu, σσ = np.meshgrid(μu, σ)
if True:
    σ, μu, μd, μe, ω = σμω(σσ, μuμu, neutral, μside, neglecte)
    ωc = ω.reshape(-1)
    μuc = μu.reshape(-1)
    σc = σ.reshape(-1)
    utils.writecols([σc, μuc, ωc / fπ**4], ["sigma", "muu", "omega"], "data/lsmpot.dat", skipevery=np.shape(ω)[1])
    σ, μu, μd, μe = σ[:,0], μu[0,:], μd[0,:], μe[0,:]
    plt.plot(σ, ω / fπ**4, "-k")
ret = σμω0(μu, neutral, μside, neglecte)
σ0, μu0, μd0, μe0, ω0 = ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4]
pti = np.argmax(np.abs(σ0[1:]-σ0[:-1])) + 1 # pti is now first point after phase transition

def incres(arr, i1, i2, n):
    arrx = np.linspace(arr[i1], arr[i2], n)
    arr = np.concatenate([arr[:i1], arrx, arr[i2:]])
    return arr

if np.abs(σ0[pti]-σ0[pti-1]) < 50:
    print("no pt")
    pti = 0 # no phase transition
else: # TODO: add more points in-between phase transition
    if False:
        μux = np.linspace(μu0[pti-1], μu0[pti], 10)
        μu = np.insert(μu, pti, μux)
        ret = σμω0(μu, neutral, μside) # recalculate everything with the new points
        σ0, μu0, μd0, μe0, ω0 = ret[:,0], ret[:,1], ret[:,2], ret[:,3], ret[:,4]
    if False:
        σ0 = incres(σ0, pti-1, pti, 10)
        μu0 = incres(μu0, pti-1, pti, 10)
        μd0 = incres(μd0, pti-1, pti, 10)
        μe0 = incres(μe0, pti-1, pti, 10)
        ω0 = incres(ω0, pti-1, pti, 10)

# write equation of state
"""
print(f"phase transition between μu = {μu[pti-1]}, {μu[pti]}")
cols = np.array([σ0, μu0, μd0, μe0, ω0 / fπ**4])
print(cols.shape)
cols = np.concatenate([cols[:,:pti], np.full((len(cols),1), np.nan), cols[:,pti:]], axis=1) # add nan at phase transition
print(cols.shape)
utils.writecols(cols, ["sigma", "muu", "mud", "mue", "omega"], "data/lsmpot_min.dat")
"""

plt.plot(σ0, ω0 / fπ**4, "-ro")
plt.show()

# plot minimum line
plt.plot(μu0, σ0, "-k.")
plt.xlabel(r"$\mu_u$")
plt.ylabel(r"$\sigma$")
plt.show()

# show densities
# TODO: is it correct that the electron density so extremely low?
nu0, nd0, ne0 = n(μu0, mq(σ0)), n(μd0, mq(σ0)), n(μe0, me)
plt.plot(μu0, nu0, "-r.", label=r"$i=u$")
plt.plot(μu0, nd0, "-g.", label=r"$i=d$")
plt.plot(μu0, ne0, "-b.", label=r"$i=e$")
plt.xlabel(r"$\mu_u$")
plt.ylabel(r"$n(\mu_i,m_i)$")
plt.legend()
plt.show()

# equation of state
P0 = -ω0
P0 = P0 - P0[0] # subtract vacuum contribution P(fπ) TODO: correct?
ϵ0 = μe0*ne0 + μu0*nu0 + μd0*nd0 - P0
plt.plot(μu0, P0, "-r.")
plt.plot(μu0, ϵ0, "-b.")
plt.xlabel(r"$P$")
plt.xlabel(r"$\epsilon$")
plt.show()

plt.plot(P0, ϵ0, "-k.")
plt.xlabel(r"$P$")
plt.ylabel(r"$\epsilon$")
plt.show()
