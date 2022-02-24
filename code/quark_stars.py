#!/usr/bin/python3

import constants
π = constants.π
from tov import massradiusplot
import utils

import numpy as np
import sympy as sp
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

Nc  = 3
Nf  = 2
fπ  = 93
mσ  = 800
mπ  = 138
mq0 = 300
me  = 0.5

m2 = 1/2*(3*mπ**2-mσ**2)
λ  = 3/fπ**2 * (mσ**2-mπ**2)
h  = fπ * mπ**2
g  = mq0 / fπ
Λ2 = mq0**2 / np.e

# symbolic complex expression for ω and dω/dσ
σ, μu, μd, μe = sp.symbols("σ μ_u μ_d μ_e", complex=True)
mq  = g*σ
ω0  =  1/2*m2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq**4/(16*π**2)*(3/2+sp.log(Λ2/mq**2))
ωu  = -Nc/(24*π**2) * ((2*μu**2-5*mq**2)*μu*sp.sqrt(μu**2-mq**2) + 3*mq**4*sp.asinh(sp.sqrt(μu**2/mq**2-1)))
ωd  = -Nc/(24*π**2) * ((2*μd**2-5*mq**2)*μd*sp.sqrt(μd**2-mq**2) + 3*mq**4*sp.asinh(sp.sqrt(μd**2/mq**2-1)))
ωe  =  -1/(24*π**2) * ((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2) + 3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))
ωc  = ω0 + ωu + ωd + ωe
dωc = sp.diff(ωc, σ)

# numeric real expressions for ω and dω/dσ
ωcf  = sp.lambdify((σ, μu, μd, μe),  ωc, "numpy") # complex function  ωcf(σ, μu, μd, μe)
dωcf = sp.lambdify((σ, μu, μd, μe), dωc, "numpy") # complex function dωcf(σ, μu, μd, μe)
def  ωf(σ, μu, μd, μe): return np.real( ωcf(σ+0j, μu+0j, μd+0j, μe+0j)) # real function ωf(σ, μu, μd, μe)
def dωf(σ, μu, μd, μe): return np.real(dωcf(σ+0j, μu+0j, μd+0j, μe+0j)) # real function ωf(σ, μu, μd, μe)

def qf(σ, μu, μd, μe):
    mq = g*σ
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-mq**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-mq**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    return +2/3*nu - 1/3*nd - 1*ne

def eos(μ, B=0, name="ϵ", outfile="", plot=False, verbose=False):
    σ, μu, μd, μe, ω = np.empty_like(μ), np.empty_like(μ), np.empty_like(μ), np.empty_like(μ), np.empty_like(μ)

    for i in range(0, len(μ)):
        μ0 = μ[i]
        def system(σ_μe): # solve system {dω == 0, q == 0}
            σ, μe = σ_μe # unpack 2 variables
            μu = μ0 - 1/2*μe
            μd = 2*μ0 - μu
            dω = dωf(σ, μu, μd, μe)
            q  =  qf(σ, μu, μd, μe)
            return (dω, q)
        guess = (σ[i-1], μe[i-1]) if i > 0 else (fπ, 0) # guess root with previous solution (if any)
        sol = scipy.optimize.root(system, guess, method="hybr")
        assert sol.success, f"{sol.message} (μ = {μ0})"
        σ0, μe0 = sol.x
        μu0 = μ0 - 1/2*μe0
        μd0 = 2*μ0 - μu0
        ω0 = ωf(σ0, μu0, μd0, μe0)
        σ[i], μu[i], μd[i], μe[i], ω[i] = σ0, μu0, μd0, μe0, ω0
        if verbose: print(f"μ = {μ0} -> σ = {σ0}, μu = {μu0}, μd = {μd0}, μe = {μe0} -> ω = {ω0}")

    P = -(ω - ω[0])
    mq = g*σ
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-mq**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-mq**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    ϵ  = -P + μu*nu + μd*nd + μe*ne

    # TODO: bag constant
    nB = 1/3 * (nu + nd)
    ϵm = 0 + μu*nu + μd*nd + μe*ne # TODO: is this what "P = 0" means?
    #plt.plot(P, (ϵ+P)/nB-930)
    #plt.show()
    Bmin = scipy.optimize.root_scalar(scipy.interpolate.interp1d(P, (ϵ+P)/nB - 930), bracket=(0, 1e9), method="brentq").root
    print(f"B^(1/4) > {Bmin**(1/4)}")
    if B < Bmin:
        print(f"WARNING: requested bag constant B = {B} is less than the minimum stable value B = {Bmin}")
    ϵ += B
    P -= B

    # convert interesting quantities to SI units
    nu *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
    nd *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
    ne *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
    P  *= constants.MeV**4 / (constants.ħ * constants.c)**3 # now in units kg*m^2/s^2/m^3
    ϵ  *= constants.MeV**4 / (constants.ħ * constants.c)**3 # now in units kg*m^2/s^2/m^3

    # interpolate dimensionless EOS
    P0 = P / constants.ϵ0 # now in TOV-dimensionless units
    ϵ0 = ϵ / constants.ϵ0 # now in TOV-dimensionless units
    print(f"interpolation range: {P0[0]} < P0 < {P0[-1]}")
    ϵint = scipy.interpolate.interp1d(P0, ϵ0)
    ϵint.__name__ = name

    # convert interesting quantities to appropriate units
    nu *= constants.fm**3                 # now in units 1/fm^3
    nd *= constants.fm**3                 # now in units 1/fm^3
    ne *= constants.fm**3                 # now in units 1/fm^3
    P  *= constants.fm**3 / constants.GeV # now in units GeV/fm^3
    ϵ  *= constants.fm**3 / constants.GeV # now in units GeV/fm^3

    if outfile != "":
        cols  = (μ, σ, μu, μd, μe, nu, nd, ne, ϵ, P)
        heads = ("mu", "sigma", "muu", "mud", "mue", "nu", "nd", "ne", "epsilon", "P")
        utils.writecols(cols, heads, outfile)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        ax1.set_xlabel(r"$\mu_b$")
        ax1.set_ylabel(r"$\sigma$")
        ax1.plot(μ, σ, "-k.")
        ax1.plot(μ, μu, "-r.")
        ax1.plot(μ, μd, "-g.")
        ax1.plot(μ, μe, "-b.")
        ax2.set_xlabel(r"$\mu_b$")
        ax2.set_ylabel(r"$n$")
        ax2.plot(μ, nu, "-r.")
        ax2.plot(μ, nd, "-g.")
        ax2.plot(μ, ne, "-b.")
        ax3.set_xlabel(r"$P$")
        ax3.set_ylabel(r"$\epsilon$")
        ax3.plot(P, ϵ, "-k.")
        plt.show()
    
    return ϵint

if __name__ == "__main__":
    # plot ω(σ, μu=μd=μ, 0)
    σ = np.linspace(-150, +150, 100)
    μ = np.linspace(0, 500, 50)
    ω = np.array([ωf(σ, μ, μ, 0) for μ in μ])
    σ0 = np.empty_like(μ)
    ω0 = np.empty_like(μ)
    for i in range(0, len(μ)):
        μ0 = μ[i]
        sol = scipy.optimize.minimize_scalar(ωf, bounds=(0, 100), args=(μ0, μ0, 0), method="bounded")
        assert sol.success, f"{sol.message} (μ = {μ0})"
        σ0[i] = sol.x
        ω0[i] = ωf(σ0[i], μ0, μ0, 0)
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\omega$")
    plt.plot(σ, ω.T, "-k")
    plt.plot(σ0, ω0, "-r.")
    plt.show()
    σc, μc, ωc = [], [], []
    for i in range(0, len(σ)):
        for j in range(0, len(μ)):
            μc.append(μ[j])
            σc.append(σ[i])
            ωc.append(ω[j,i])
    utils.writecols([μc, σc, list(np.array(ωc)/100**4), μc, list(σ0), list(ω0/100**4)], ["mu", "sigma", "omega", "mu0", "sigma0", "omega0"], "data/2flavpot.dat", skipevery=len(μ))

    # plot equation of state
    μ = np.linspace(0, 1000, 250)[1:]
    ϵ = eos(μ, B=0, plot=True, outfile="data/2flaveos.dat", verbose=True)

    # solve TOV equation for different bag pressures
    μ = np.concatenate([np.linspace(0, 600, 200)[1:], np.linspace(600, 5000, 100)])
    opts = { "tolD": 0.25, "maxdr": 1e-2, "visual": False }
    for B14 in [27, 34, 41, 48, 55, 62, 69]:
        outfile = f"data/quarkstar2f_B14_{B14}.dat"
        print(f"B = {B14}^4, outfile = {outfile}")
        ϵ = eos(μ, B=B14**4, name=f"ϵ2f", plot=False, verbose=False)
        massradiusplot(ϵ, (1e-5, 1e1), **opts, nmodes=3, outfile=outfile)
