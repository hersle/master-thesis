#!/usr/bin/python3

import constants
π = constants.π
from tov import massradiusplot, soltov
import utils

import numpy as np
import sympy as sp
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

fπ = 93
fK = 113

Nc = 3
Nf = 3
m2 = -491.7**2
λ1 = -6.19
λ2 = 85.3
hx = 121.0**3
hy = 336.4**3
g  = 6.45 # TODO: update text with 6.45
Λ2 = 208.0**2
me = 0.5

# symbolic complex expression for ω and dω/dσ
σx, σy, μu, μd, μs, μe = sp.symbols("σ_x σ_y μ_u μ_d μ_s μ_e", complex=True)
mu, md, ms = g*σx/2, g*σx/2, g*σy/np.sqrt(2)
ω0  = m2/2*(σx**2+σy**2) + λ1/4*(σx**2+σy**2)**2 + λ2/8*(σx**4+2*σy**4) - hx*σx - hy*σy 
ωr  = Nc/(16*π**2)*(mu**4*(3/2+sp.log(Λ2/mu**2))+md**4*(3/2+sp.log(Λ2/md**2))+ms**4*(3/2+sp.log(Λ2/ms**2)))
ωu  = -Nc/(24*π**2)*((2*μu**2-5*mu**2)*μu*sp.sqrt(μu**2-mu**2)
                     +3*mu**4*sp.asinh(sp.sqrt(μu**2/mu**2-1)))
ωd  = -Nc/(24*π**2)*((2*μd**2-5*md**2)*μd*sp.sqrt(μd**2-md**2)
                     +3*md**4*sp.asinh(sp.sqrt(μd**2/md**2-1)))
ωs  = -Nc/(24*π**2)*((2*μs**2-5*ms**2)*μs*sp.sqrt(μs**2-ms**2)
                     +3*ms**4*sp.asinh(sp.sqrt(μs**2/ms**2-1)))
ωe  =  -1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)
                     +3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))
ωc  = ω0 + ωr + ωu + ωd + ωs + ωe
dωx = sp.diff(ωc, σx)
dωy = sp.diff(ωc, σy)

# numeric complex functions ωcf(σ, μu, μd, μe) and dωcf(σ, μu, μd, μe)
ωcf  = sp.lambdify((σx, σy, μu, μd, μs, μe),  ωc, "numpy")
dωxc = sp.lambdify((σx, σy, μu, μd, μs, μe), dωx, "numpy")
dωyc = sp.lambdify((σx, σy, μu, μd, μs, μe), dωy, "numpy")

print(dωx)

# numeric real functions ωf(σ, μu, μd, μe) and dωf(σ, μu, μd, μe)
def   ωf(σx, σy, μu, μd, μs, μe): return np.real( ωcf(σx+0j, σy+0j, μu+0j, μd+0j, μs+0j, μe+0j))
def dωxf(σx, σy, μu, μd, μs, μe): return np.real(dωxc(σx+0j, σy+0j, μu+0j, μd+0j, μs+0j, μe+0j))
def dωyf(σx, σy, μu, μd, μs, μe): return np.real(dωyc(σx+0j, σy+0j, μu+0j, μd+0j, μs+0j, μe+0j))

def qf(σx, σy, μu, μd, μs, μe):
    mu, md, ms = g*σx/2, g*σx/2, g*σy/np.sqrt(2)
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-mu**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-md**2+0j, 3/2))
    ns = Nc/(3*π**2) * np.real(np.power(μs**2-ms**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    return +2/3*nu - 1/3*nd - 1/3*ns - 1*ne

def eos(μ, B=None, interaction=True, name="ϵ", outfile="", plot=False, nint=False, verbose=False):
    if interaction:
        σx = np.empty_like(μ)
        σy = np.empty_like(μ)
        μu = np.empty_like(μ)
        μd = np.empty_like(μ)
        μs = np.empty_like(μ)
        μe = np.empty_like(μ)
        ω  = np.empty_like(μ)

        for i in range(0, len(μ)):
            μ0 = μ[i]
            def system(σx_σy_μe):
                σx0, σy0, μe0 = σx_σy_μe # unpack variables
                μu0 = μ0 - μe0/2
                μd0 = μu0 + μe0
                μs0 = μd0
                dωx = dωxf(σx0, σy0, μu0, μd0, μs0, μe0)
                dωy = dωyf(σx0, σy0, μu0, μd0, μs0, μe0)
                q  =    qf(σx0, σy0, μu0, μd0, μs0, μe0)
                return (dωx, dωy, q)
            guess = (σx[i-1], σy[i-1], μe[i-1]) if i > 0 else (fπ, np.sqrt(2)*fK-fπ/np.sqrt(2), 0) # use previous solution
            sol = scipy.optimize.root(system, guess, method="lm") # lm works, hybr works but unstable for small μ
            assert sol.success, f"{sol.message} (μ = {μ0})"
            σx0, σy0, μe0 = sol.x
            μu0 = μ0 - μe0/2
            μd0 = μu0 + μe0
            μs0 = μd0
            ω0 = ωf(σx0, σy0, μu0, μd0, μs0, μe0)
            σx[i], σy[i], μu[i], μd[i], μs[i], μe[i], ω[i] = σx0, σy0, μu0, μd0, μs0, μe0, ω0
            if verbose:
                print(f"μ = {μ0}, σx = {σx0}, σy = {σy0}, μu = {μu0}, μd = {μd0}, μs = {μs0}, μe = {μe0} -> ω = {ω0}")
    else:
        # analytical solution
        #me = 0 # zero electron mass (override global variable)
        pass # TODO:
        """
        μu = 2/(1+2**(1/3)) * μ
        μd = 2*μ - μu
        μe = μd - μu # (2**(1/3)-1) * μu # close to μu0/4
        ωu = -1/(12*π**2) * (Nc*μu**4)
        ωd = -1/(12*π**2) * (Nc*μd**4)
        ωe =  -1/(24*π**2)*((2*μe**2-5*me**2)*μe*np.sqrt(μe**2-me**2)
                            +3*me**4*np.arcsinh(np.sqrt(μe**2/me**2-1)))
        ω = ωu + ωd + ωe
        σ = np.zeros_like(μ) # zero quark mass
        """

    print(f"ω[0] = {ω[0]}")

    if interaction:
        P = -(ω - ω[0])
    else:
        pass # TODO:
        #P = -ω - 599805285.843199
    mu, md, ms = g*σx/2, g*σx/2, g*σy/np.sqrt(2)
    nu = Nc/(3*π**2) * np.real(np.power(μu**2-mu**2+0j, 3/2))
    nd = Nc/(3*π**2) * np.real(np.power(μd**2-md**2+0j, 3/2))
    ns = Nc/(3*π**2) * np.real(np.power(μs**2-ms**2+0j, 3/2))
    ne =  1/(3*π**2) * np.real(np.power(μe**2-me**2+0j, 3/2))
    ϵ  = -P + μu*nu + μd*nd + + μs*ns + μe*ne

    # TODO: bag constant
    if B is not None:
        nB = 1/3 * (nu + nd + ns)
        ϵB = 0 + μu*nu + μd*nd + μs*ns # TODO: is this what "P = 0" means?
        f = scipy.interpolate.interp1d(P, ϵB/nB - 930)
        Bmin = scipy.optimize.root_scalar(f, bracket=(0, 1e9), method="brentq").root
        print(f"bag constant upper bound: B^(1/4) < {Bmin**(1/4)} MeV")
        ϵ += B
        P -= B

    # convert interesting quantities to SI units
    nu *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
    nd *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
    ns *= constants.MeV**3 / (constants.ħ * constants.c)**3 # now in units 1/m^3
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
    ns *= constants.fm**3                 # now in units 1/fm^3
    ne *= constants.fm**3                 # now in units 1/fm^3
    P  *= constants.fm**3 / constants.GeV # now in units GeV/fm^3
    ϵ  *= constants.fm**3 / constants.GeV # now in units GeV/fm^3

    if outfile != "":
        cols  = (μ, σx, σy, μu, μd, μs, μe, nu, nd, ns, ne, ϵ, P)
        heads = ("mu", "sigmax", "sigmay", "muu", "mud", "mus", "mue", "nu", "nd", "ns", "ne", "epsilon", "P")
        utils.writecols(cols, heads, outfile)

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
        ax1.set_xlabel(r"$\mu_b$")
        ax1.plot(μ, σx, color="orange")
        ax1.plot(μ, σy, color="yellow")
        ax1.plot(μ, μu, color="red")
        ax1.plot(μ, μd, color="green")
        ax1.plot(μ, μs, color="purple")
        ax1.plot(μ, μe, color="blue")
        ax2.set_xlabel(r"$\mu_b$")
        ax2.set_ylabel(r"$n$")
        ax2.plot(μ, nu, color="red")
        ax2.plot(μ, nd, color="green")
        ax2.plot(μ, ns, color="purple")
        ax2.plot(μ, ne, color="blue")
        ax3.set_xlabel(r"$P$")
        ax3.set_ylabel(r"$\epsilon$")
        ax3.plot(P, ϵ, "-k.")
        plt.show()

    if nint:
        nuint = scipy.interpolate.interp1d(P0, nu)
        ndint = scipy.interpolate.interp1d(P0, nd)
        nsint = scipy.interpolate.interp1d(P0, ns)
        neint = scipy.interpolate.interp1d(P0, ne)
        return ϵint, nuint, ndint, nsint, neint
    else:
        return ϵint

if __name__ == "__main__":
    # plot massive, interacting and massless, free equation of state
    #μ = np.linspace(0, 1000, 1000)[1:]
    #ϵ = eos(μ, plot=True, verbose=True, outfile="data/3flaveos.dat")

    # solve TOV equation for different bag pressures
    μ = np.concatenate([np.linspace(0, 700, 200)[1:], np.linspace(700, 5000, 100)])
    opts = { "tolD": 0.125, "maxdr": 1e-2, "visual": False }

    # with radial density plots
    B14 = 38
    Pcs = [0.0006, 0.0008, 0.001]
    cols = [Pcs]
    heads = ["Pcs"]
    for i, Pc in enumerate(Pcs):
        ϵ, nu, nd, ns, ne = eos(μ, B=B14**4, nint=True)
        rs, ms, Ps, αs, ϵs = soltov(ϵ, Pc, maxdr=opts["maxdr"])
        nus, nds, nss, nes = nu(Ps), nd(Ps), ns(Ps), ne(Ps)

        heads += [f"r{i}", f"P{i}", f"nu{i}", f"nd{i}", f"ns{i}", f"ne{i}"]
        cols += [list(rs), list(Ps), list(nus), list(nds), list(nss), list(nes)]

        """
        plt.plot(rs, Ps, "-k.")
        plt.show()
        plt.plot(rs, nu(Ps), "-r.")
        plt.plot(rs, nd(Ps), "-g.")
        plt.plot(rs, ns(Ps), "-b.")
        plt.show()
        """
    utils.writecols(cols, heads, f"data/quarkstar3f_B14_{B14}_densities.dat")

    # shooting method crashes for B14 >= 139
    for B14 in [6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132]:
        outfile = f"data/quarkstar3f_B14_{B14}.dat"
        print(f"B = {B14}^4, outfile = {outfile}")
        ϵ = eos(μ, B=B14**4, name=f"ϵ3f", plot=False, verbose=False)
        massradiusplot(ϵ, (1e-7, 1e1), **opts, nmodes=0, outfile=outfile)
