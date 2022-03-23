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

Nc = 3
Nf = 2
fπ = 93
mσ = 800
mπ = 138
mq = 300
me = 0.5

# symbolic complex expression for ω and dω/dσ
Δ, μu, μd, μe = sp.symbols("Δ μ_u μ_d μ_e", complex=True)
def r(p2): return sp.sqrt(4*mq**2/p2-1)
def F(p2): return 2 - 2*r(p2)*sp.atan(1/r(p2))
def dF(p2): return 4*mq**2*r(p2)/(p2*(4*mq**2-r(p2)**2))*sp.atan(1/r(p2))-1/p2
ωc = (3/4*mπ**2*fπ**2*(1-4*mq**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ**2 / mq**2 -
      mσ**2*fπ**2/4*(1+4*mq**2*Nc/(16*π**2*fπ**2)*((1-4*mq**2/mσ**2)*F(mσ**2)+4*mq**2/mσ**2-F(mπ**2)-mπ**2*dF(mπ**2))) * Δ**2/mq**2 +
      mσ**2*fπ**2/8*(1-4*mq**2*Nc/(16*π**2*fπ**2)*(4*mq**2/mσ**2*sp.log(Δ**2/mq**2)-(1-4*mq**2/mσ**2)*F(mσ**2)+F(mπ**2)+mπ**2*dF(mπ**2)))* Δ**4 / mq**4 -
      mπ**2*fπ**2/8*(1-4*mq**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ**4/mq**4 -
      mπ**2*fπ**2*(1-4*mq**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ/mq +
      3*Nc/(16*π**2) * Δ**4 -
      Nc/(24*π**2)*((2*μu**2-5*Δ**2)*μu*sp.sqrt(μu**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μu**2/Δ**2-1))) -
      Nc/(24*π**2)*((2*μd**2-5*Δ**2)*μd*sp.sqrt(μd**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μd**2/Δ**2-1))) -
      1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1))))
dωc = sp.diff(ωc, Δ)

ωcf  = sp.lambdify((Δ, μu, μd, μe),  ωc, "numpy")
dωcf = sp.lambdify((Δ, μu, μd, μe), dωc, "numpy")

def  ωf(Δ, μu, μd, μe): return np.real( ωcf(Δ+0j, μu+0j, μd+0j, μe+0j))
def dωf(Δ, μu, μd, μe): return np.real(dωcf(Δ+0j, μu+0j, μd+0j, μe+0j))

def qf(Δ, μu, μd, μe):
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-Δ**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-Δ**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    return +2/3*nu - 1/3*nd - 1*ne

def eos(μ, B=None, interaction=True, name="ϵ", outfile="", plot=False, verbose=False):
    Δ  = np.empty_like(μ)
    μu = np.empty_like(μ)
    μd = np.empty_like(μ)
    μe = np.empty_like(μ)
    ω  = np.empty_like(μ)

    for i in range(0, len(μ)):
        μ0 = μ[i]
        def system(Δ_μe): # solve system {dω == 0, q == 0}
            Δ, μe = Δ_μe # unpack 2 variables
            μu = μ0 - 1/2*μe
            μd = 2*μ0 - μu
            dω = dωf(Δ, μu, μd, μe)
            q  =  qf(Δ, μu, μd, μe)
            return (dω, q)
        guess = (Δ[i-1], μe[i-1]) if i > 0 else (300, 0) # use previous solution
        sol = scipy.optimize.root(system, guess, method="hybr")
        assert sol.success, f"{sol.message} (μ = {μ0})"
        Δ0, μe0 = sol.x
        μu0 = μ0 - 1/2*μe0
        μd0 = 2*μ0 - μu0
        ω0 = ωf(Δ0, μu0, μd0, μe0)
        Δ[i], μu[i], μd[i], μe[i], ω[i] = Δ0, μu0, μd0, μe0, ω0
        if verbose:
            print(f"μ = {μ0}, Δ = {Δ0}, μu = {μu0}, μd = {μd0}, μe = {μe0} -> ω = {ω0}")

    print(f"ω[0] = {ω[0]}")

    P = -(ω - ω[0])
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-Δ**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-Δ**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    ϵ  = -P + μu*nu + μd*nd + μe*ne

    # TODO: bag constant
    if B is not None:
        nB = 1/3 * (nu + nd)
        ϵB = 0 + μu*nu + μd*nd # TODO: is this what "P = 0" means?
        f = scipy.interpolate.interp1d(P, ϵB/nB - 930)
        Bmin = scipy.optimize.root_scalar(f, bracket=(0, 1e9), method="brentq").root
        print(f"bag constant lower bound: B^(1/4) > {Bmin**(1/4)} MeV")
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
        cols  = (μ, Δ, μu, μd, μe, nu, nd, ne, ϵ, P)
        heads = ("mu", "Delta", "muu", "mud", "mue", "nu", "nd", "ne", "epsilon", "P")
        utils.writecols(cols, heads, outfile)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        ax1.set_xlabel(r"$\mu_b$")
        ax1.set_ylabel(r"$\sigma$")
        ax1.plot(μ, Δ, "-k.")
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
    # plot massive, interacting and massless, free equation of state
    μ = np.linspace(0, 1000, 1000)[1:]
    ϵ = eos(μ, plot=True, outfile="data/2flavrefeos.dat", verbose=True)

    # solve TOV equation for different bag pressures
    μ = np.concatenate([np.linspace(0, 600, 200)[1:], np.linspace(600, 5000, 100)])
    opts = { "tolD": 0.25, "maxdr": 1e-2, "visual": False }
    for B14 in [27, 34, 41, 48, 55, 62, 69]:
        outfile = f"data/quarkstar2f_ref_B14_{B14}.dat"
        print(f"B = {B14}^4, outfile = {outfile}")
        ϵ = eos(μ, B=B14**4, name=f"ϵ2f", plot=False, verbose=False)
        massradiusplot(ϵ, (1e-7, 1e1), **opts, nmodes=0, outfile=outfile)
