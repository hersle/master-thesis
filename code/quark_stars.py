#!/usr/bin/python3

from constants import π, ħ, c, ϵ0, MeV, GeV, fm
from tov import massradiusplot, soltov
import utils

import numpy as np
import sympy as sp
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import mayavi.mlab as mlab

Nc = 3
fπ = 93
fK = 113
muf = 5 # current/lone/free masses (i.e. without gluons)
mdf = 7
msf = 150
mu0 = 300 # constituent masses (i.e. with gluons)
md0 = mu0
ms0 = 429 # only used for root equation guess
mσ = 800
mπ = 138
mK = 496
ma0 = 1028.7
mη = 539
mηp = 963
me = 0.5

nsat = 0.165

σx0 = fπ
σy0 = np.sqrt(2)*fK-fπ/np.sqrt(2)

tovopts = {"tolD": 0.01, "maxdr": 1e-2, "nmodes": 0}
tovμQ = np.concatenate([np.linspace(0, 700, 200)[1:], np.linspace(700, 5000, 100)])

def charge(mu, md, ms, μu, μd, μs, μe):
    nu = Nc/(3*π**2) * np.real((μu**2-mu**2+0j)**(3/2))
    nd = Nc/(3*π**2) * np.real((μd**2-md**2+0j)**(3/2))
    ns = Nc/(3*π**2) * np.real((μs**2-ms**2+0j)**(3/2))
    ne =  1/(3*π**2) * np.real((μe**2-me**2+0j)**(3/2))
    return +2/3*nu - 1/3*nd - 1/3*ns - 1*ne

# solve μQ=(μu+μd)/2, μd=μu+μe, μs=μd for (μu,μd,μs)
def μelim(μQ, μe):
    μu = μQ - μe/2
    μd = μu + μe
    μs = μd
    return μu, μd, μs

class Model:
    def __init__(self, name, mσ=mσ, mπ=mπ, mK=mK, ma0=ma0):
        self.name = name
        self.mu, self.md, self.ms = self.vacuum_masses()
        print(f"Meson masses: mσ = {mσ:.1f} MeV, mπ = {mπ:.1f}, mK = {mK:.1f}")
        print(f"Quark masses: mu = md = {self.mu:.1f} MeV, ms = {self.ms:.1f} MeV")
        self.mσ = mσ
        self.mπ = mπ
        self.mK = mK
        self.ma0 = ma0

    def eos(self, B=0, N=1000, plot=False, write=False, debugmaxwell=False):
        mu, md, ms, μu, μd, μs, μe = self.eossolve(N=N) # model-dependent function

        # extend solutions to μ = 0 to show Silver-Blaze property
        μu = np.insert(μu, 0, 0)
        μd = np.insert(μd, 0, 0)
        μs = np.insert(μs, 0, 0)
        μe = np.insert(μe, 0, 0)
        mu = np.insert(mu, 0, mu[0])
        md = np.insert(md, 0, md[0])
        ms = np.insert(ms, 0, ms[0])

        μQ = (μu + μd) / 2

        Ω = self.Ω(mu, md, ms, μu+0j, μd+0j, μs+0j, μe+0j) # model-dependent
        P, P0 = -Ω, -Ω[0]
        P = P - P0

        nu = Nc/(3*π**2) * np.real((μu**2-mu**2+0j)**(3/2))
        nd = Nc/(3*π**2) * np.real((μd**2-md**2+0j)**(3/2))
        ns = Nc/(3*π**2) * np.real((μs**2-ms**2+0j)**(3/2))
        ne =  1/(3*π**2) * np.real((μe**2-me**2+0j)**(3/2))
        ϵ  = -P + μu*nu + μd*nd + μs*ns + μe*ne

        # print bag constant bound (upper or lower, depending on circumstances)
        nB = 1/3*(nu+nd+ns)
        def EperB(B):
            PB = P - B
            ϵB = ϵ + B
            return np.interp(0, PB, ϵB/nB) # at P=0
        f = lambda B: EperB(B) - 930 
        Bs = np.linspace(0, 300, 10000)**4
        if plot:
            plt.plot(Bs**(1/4), [f(Bs) for Bs in Bs])
            plt.ylim(-500, +500)
            plt.show()
        try:
            # note: bracket lower bound is sensitive
            sol = scipy.optimize.root_scalar(f, method="brentq", bracket=(1e5, 300**4))
            assert sol.converged
            Bbound = sol.root
        except ValueError:
            print("alternative bag bound method")
            Bs = np.linspace(0, 300, 10000)**4
            Bbound = Bs[np.argmin([f(B) for B in Bs])]
        print(f"Bag constant bound: B^(1/4) = {Bbound**(1/4)} MeV", end=" ")
        print(f"({(P0+Bbound)**(1/4)} MeV)")

        P -= B
        ϵ += B

        if plot:
            plt.plot(P, ϵ/nB, ".-k")
            plt.axhline(930, color="red")
            plt.show()

        # plot bag pressure
        if plot:
            μvac = np.full_like(mu, 0+0j)
            PB = -self.Ω(mu, md, ms, μvac, μvac, μvac, μvac) - P0 - B
            PQ = P - PB
            plt.plot((μu+μd)/2, np.sign(PB)*np.abs(PB)**(1/4), label="bag")
            plt.plot((μu+μd)/2, np.sign(PQ)*np.abs(PQ)**(1/4), label="quark")
            plt.plot((μu+μd)/2, np.sign(P)*np.abs(P)**(1/4), label="total")
            plt.xlabel(r"$\mu$")
            plt.ylabel(r"$P$")
            plt.legend()
            plt.show()

        P1 = P[0]
        i2 = np.argmax(np.gradient(P) < 0) # last index of increasing pressure
        P2 = P[i2]
        have_phase_transition = (i2 != 0) and (P2 > 0)
        print("Phase transition?", have_phase_transition)
        Porg, ϵorg = np.copy(P), np.copy(ϵ) # save pre-Maxwell-construction P, ϵ
        if have_phase_transition:
            i3 = i2 + np.argmax(np.gradient(P[i2:]) > 0) - 1
            P3 = P[i3]

            # debug Maxwell construction
            if debugmaxwell:
                plt.plot(P[:i2+1], ϵ[:i2+1], marker=".", color="red")
                plt.plot(P[i2:i3+1], ϵ[i2:i3+1], marker=".", color="green")
                plt.plot(P[i3:], ϵ[i3:], marker=".", color="blue")
                plt.show()

            def gibbs_area(Pt):
                j1 = np.argmax(P[:i2+1] >= Pt) # first pt on 1-curve with greater P
                j2 = i3 + np.argmax(P[i3:] >= Pt) # first pt on 2-curve with greater P
                ϵ1 = np.interp(Pt, P[:i2+1], ϵ[:i2+1])
                ϵ2 = np.interp(Pt, P[i3:], ϵ[i3:])
                P12 = np.concatenate([[Pt], P[j1:j2], [Pt]])
                ϵ12 = np.concatenate([[ϵ1], ϵ[j1:j2], [ϵ2]])
                ret = np.trapz(1/ϵ12, P12)
                print(f"gibbs_area({Pt}) = {ret}")
                return ret

            # find P that gives zero Gibbs area
            # (pray that P[0]+1e-3 works, since P[0] = 0 gives 0-div error
            sol = scipy.optimize.root_scalar(gibbs_area, bracket=(P[0]+1e-3, P[i2]), \
                                             method="brentq")
            assert sol.converged
            Pt = sol.root
            print(f"Phase transition pressure: {Pt} MeV^4")

            if debugmaxwell:
                plt.plot(1/ϵ, P, color="gray")
                plt.ylim(1.1*np.min(P), -1.1*np.min(P))

            j1 = np.argmax(P[:i2+1] >= Pt) # first pt on 1-curve with greater P
            j2 = i3 + np.argmax(P[i3:] >= Pt) # first pt on 2-curve with greater P
            ϵ1 = np.interp(Pt, P[:i2+1], ϵ[:i2+1])
            ϵ2 = np.interp(Pt, P[i3:], ϵ[i3:])

            # fix array by only modifying EOS,
            # but fill out with points in phase transition
            ϵ1 = np.interp(Pt, P[:i2+1], ϵ[:i2+1])
            ϵ2 = np.interp(Pt, P[i3:], ϵ[i3:])
            Ntarget = len(mu)
            Nnow = len(ϵ[:j1]) + len(ϵ[j2:])
            Nadd = Ntarget - Nnow
            ϵ = np.concatenate((ϵ[:j1], np.linspace(ϵ1, ϵ2, Nadd), ϵ[j2:]))
            P = np.concatenate((P[:j1], np.linspace(Pt, Pt, Nadd), P[j2:]))

            if debugmaxwell:
                plt.plot(1/ϵ, P, color="black")
                plt.show()

        # convert interesting quantities to SI units
        nu *= MeV**3 / (ħ*c)**3 * fm**3 # now in units 1/fm^3
        nd *= MeV**3 / (ħ*c)**3 * fm**3 # now in units 1/fm^3
        ns *= MeV**3 / (ħ*c)**3 * fm**3 # now in units 1/fm^3
        ne *= MeV**3 / (ħ*c)**3 * fm**3 # now in units 1/fm^3
        P0 *= MeV**4 / (ħ*c)**3 # now in units kg*m^2/s^2/m^3
        P  *= MeV**4 / (ħ*c)**3 # now in units kg*m^2/s^2/m^3
        ϵ  *= MeV**4 / (ħ*c)**3 # now in units kg*m^2/s^2/m^3
        Porg *= MeV**4 / (ħ*c)**3 # now in units kg*m^2/s^2/m^3
        ϵorg *= MeV**4 / (ħ*c)**3 # now in units kg*m^2/s^2/m^3

        # convert interesting quantities to appropriate units
        P0 *= fm**3 / GeV # now in units GeV/fm^3
        P  *= fm**3 / GeV # now in units GeV/fm^3
        ϵ  *= fm**3 / GeV # now in units GeV/fm^3
        Porg *= fm**3 / GeV # now in units GeV/fm^3
        ϵorg *= fm**3 / GeV # now in units GeV/fm^3

        print(f"P0 = {P0}")

        if write:
            cols  = [mu, md, ms, μu, μd, μs, μe, nu, nd, ns, ne, ϵ, P, ϵorg, Porg]
            heads = ["mu", "md", "ms", "muu", "mud", "mus", "mue", \
                     "nu", "nd", "ns", "ne", "epsilon", "P", "epsilonorg", "Porg"]
            outfile = f"data/{self.name}/eos_sigma_{self.mσ}.dat"
            utils.writecols(cols, heads, outfile)

        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(25, 5))

            ax1.set_xlabel(r"$\mu_Q$")
            ax1.plot(μQ, mu, ".-", color="orange", label=r"$m_u$")
            ax1.plot(μQ, md, ".-", color="orange", label=r"$m_d$")
            ax1.plot(μQ, ms, ".-", color="yellow", label=r"$m_s$")
            ax1.plot(μQ, μu, ".-", color="red", label=r"$\mu_u$")
            ax1.plot(μQ, μd, ".-", color="green", label=r"$\mu_d$")
            ax1.plot(μQ, μs, ".-", color="purple", label=r"$\mu_s$")
            ax1.plot(μQ, μe, ".-", color="blue", label=r"$\mu_e$")
            ax1.legend()

            ax2.set_xlabel(r"$\mu_Q$")
            ax2.set_ylabel(r"$n$")
            ax2.plot(μQ, nu, ".-", color="red", label=r"$n_u$")
            ax2.plot(μQ, nd, ".-", color="green", label=r"$n_d$")
            ax2.plot(μQ, ns, ".-", color="purple", label=r"$n_s$")
            ax2.plot(μQ, ne, ".-", color="blue", label=r"$n_e$")
            ax2.legend()

            ax3.set_xlabel(r"$P$")
            ax3.set_ylabel(r"$\epsilon$")
            ax3.plot(Porg, ϵorg, ".-", color="gray") # compare
            ax3.plot(P, ϵ, ".-", color="black")

            ax4.plot(μQ, (P/(MeV**4 / (ħ*c)**3)/(fm**3 / GeV))**0.25)
            #P0 *= (MeV**4 / (ħ*c)**3) # now in units kg*m^2/s^2/m^3
            #P0 *= (fm**3 / GeV) # now in units GeV/fm^3
            ax4.set_xlabel(r"$\mu / MeV$")
            ax4.set_ylabel(r"$P^{\frac{1}{4}} / MeV$")
            ax4.set_xlim(0, 1000)
            ax4.set_ylim(0, 500)

            plt.show()

        # interpolate dimensionless EOS
        P /= (fm**3/GeV) * ϵ0 # now in TOV-dimensionless units
        ϵ /= (fm**3/GeV) * ϵ0 # now in TOV-dimensionless units
        ϵ = np.concatenate(([0, np.interp(0, P, ϵ)], ϵ[P>0]))
        nu = np.concatenate(([0, np.interp(0, P, nu)], nu[P>0]))
        nd = np.concatenate(([0, np.interp(0, P, nd)], nd[P>0]))
        ns = np.concatenate(([0, np.interp(0, P, ns)], ns[P>0]))
        ne = np.concatenate(([0, np.interp(0, P, ne)], ne[P>0]))
        μQ = np.concatenate(([0, np.interp(0, P, μQ)], μQ[P>0]))
        P = np.concatenate(([P[0] - 10, 0], P[P>0])) # avoid interp errors w/ ϵ(P<Pmin)=0
        print(f"interpolation range: {P[0]} < P < {P[-1]}")
        ϵint = scipy.interpolate.interp1d(P, ϵ); ϵint.__name__ = self.name
        nuint = scipy.interpolate.interp1d(P, nu)
        ndint = scipy.interpolate.interp1d(P, nd)
        nsint = scipy.interpolate.interp1d(P, ns)
        neint = scipy.interpolate.interp1d(P, ne)
        μQint = scipy.interpolate.interp1d(P, μQ)
        return ϵint, nuint, ndint, nsint, neint, μQint

    def star(self, Pc, B14, plot=False, write=False):
        ϵ, nu, nd, ns, ne, μQ = self.eos(B=B14**4)
        rs, ms, Ps, αs, ϵs = soltov(ϵ, Pc, maxdr=tovopts["maxdr"])
        nus, nds, nss, nes, μQs = nu(Ps), nd(Ps), ns(Ps), ne(Ps), μQ(Ps)
        xs = rs / rs[-1] # dimensionless radius [0, 1]

        Ps *= (fm**3/GeV) * ϵ0 # now in GeV/fm^3
        ϵs *= (fm**3/GeV) * ϵ0 # now in GeV/fm^3
        nus /= 0.165 # now in units of n_sat
        nds /= 0.165 # now in units of n_sat
        nss /= 0.165 # now in units of n_sat
        nes /= 0.165 # now in units of n_sat

        if plot:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            ax1.set_xlabel(r"$r$")
            ax2.set_xlabel(r"$r$")
            ax3.set_xlabel(r"$r$")
            ax4.set_xlabel(r"$r$")
            ax1.plot(rs, Ps, label=r"$P$")
            ax1.plot(rs, ϵs, label=r"$\epsilon$")
            ax2.plot(rs, ms, label=r"$m$")
            ax3.plot(rs, μQs, label=r"$\mu_Q$")
            ax4.plot(rs, nus, label=r"$n_u$")
            ax4.plot(rs, nds, label=r"$n_d$")
            ax4.plot(rs, nss, label=r"$n_s$")
            ax4.plot(rs, nes, label=r"$n_e$")
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            plt.show()

        if write:
            heads = ["r", "x", "m", "P", "epsilon", "nu", "nd", "ns", "ne", "muQ"]
            cols = [rs, xs, ms, Ps, ϵs, nus, nds, nss, nes, μQs]
            outfile = f"data/{self.name}/star_sigma_{self.mσ}_B14_{B14}_Pc_{Pc:.7f}.dat"
            utils.writecols(cols, heads, outfile)

    def stars(self, B14, P1P2, N=1000, plot=False, write=False):
        if write:
            outfile = f"data/{self.name}/stars_sigma_{self.mσ}_B14_{B14}.dat"
        else:
            outfile = ""
        print(f"B = ({B14} MeV)^4, outfile = {outfile}")
        ϵ, _, _, _, _, _ = self.eos(N=N, B=B14**4, plot=False)
        massradiusplot(ϵ, P1P2, **tovopts, visual=plot, outfile=outfile)

class MITModel(Model):
    pass

class MIT2FlavorModel(MITModel):
    def __init__(self):
        Model.__init__(self, "MIT2F")
        self.Ω = lambda mu, md, ms, μu, μd, μs, μe: np.real(
            -Nc/(24*π**2)*((2*μu**2-5*muf**2)*μu*np.sqrt(μu**2-muf**2+0j)+\
            3*muf**4*np.arcsinh(np.sqrt(μu**2/muf**2-1+0j))) + \
            -Nc/(24*π**2)*((2*μd**2-5*mdf**2)*μd*np.sqrt(μd**2-mdf**2+0j)+\
            3*mdf**4*np.arcsinh(np.sqrt(μd**2/mdf**2-1+0j))) + \
            -1/(24*π**2)*((2*μe**2-5*me**2)*μe*np.sqrt(μe**2-me**2+0j)+\
            3*me**4*np.arcsinh(np.sqrt(μe**2/me**2-1+0j)))
        )

    def vacuum_masses(self):
        return muf, mdf, 0

    def solve(self, μQ):
        def q(μe):
            μu, μd, _ = μelim(μQ, μe)
            return charge(muf, mdf, 0, μu, μd, 0, μe)
        sol = scipy.optimize.root_scalar(q, method="bisect", bracket=(0, 1e5))
        assert sol.converged, f"{sol.flag} (μQ = {μQ})"
        μe = sol.root
        μu, μd, _ = μelim(μQ, μe)
        return μu, μd, μe

    def eossolve(self, N):
        μQ = np.linspace(muf, 2000, N)
        μu = np.empty_like(μQ)
        μd = np.empty_like(μQ)
        μe = np.empty_like(μQ)

        for i in range(0, len(μQ)):
            μu[i], μd[i], μe[i] = self.solve(μQ[i])
            print(f"μQ = {μQ[i]:.2f}, μu = {μu[i]:.2f}, ", end="")
            print(f"μd = {μd[i]:.2f}, μe = {μe[i]:.2f}")

        mu = np.full_like(μQ, muf)
        md = np.full_like(μQ, mdf)
        ms = np.full_like(μQ, 0)
        μs = np.full_like(μQ, 0)
        return mu, md, ms, μu, μd, μs, μe

class MIT3FlavorModel(MITModel):
    def __init__(self):
        Model.__init__(self, "MIT3F")
        self.Ω = lambda mu, md, ms, μu, μd, μs, μe: np.real(
            -Nc/(24*π**2)*((2*μu**2-5*muf**2)*μu*np.sqrt(μu**2-muf**2)+\
            3*muf**4*np.arcsinh(np.sqrt(μu**2/muf**2-1))) + \
            -Nc/(24*π**2)*((2*μd**2-5*mdf**2)*μd*np.sqrt(μd**2-mdf**2)+\
            3*mdf**4*np.arcsinh(np.sqrt(μd**2/mdf**2-1))) + \
            -Nc/(24*π**2)*((2*μs**2-5*msf**2)*μs*np.sqrt(μs**2-msf**2)+\
            3*msf**4*np.arcsinh(np.sqrt(μs**2/msf**2-1))) + \
            -1/(24*π**2)*((2*μe**2-5*me**2)*μe*np.sqrt(μe**2-me**2)+\
            3*me**4*np.arcsinh(np.sqrt(μe**2/me**2-1)))
        )

    def vacuum_masses(self):
        return muf, mdf, msf

    def solve(self, μQ):
        def q(μe):
            μu, μd, μs = μelim(μQ, μe)
            return charge(muf, mdf, msf, μu, μd, μs, μe)
        sol = scipy.optimize.root_scalar(q, method="bisect", bracket=(0, 1e5))
        assert sol.converged, f"{sol.flag} (μQ = {μQ})"
        μe = sol.root
        μu, μd, μs = μelim(μQ, μe)
        return μu, μd, μs, μe

    def eossolve(self, N):
        μQ = np.linspace(muf, 2000, N)
        μu = np.empty_like(μQ)
        μd = np.empty_like(μQ)
        μs = np.empty_like(μQ)
        μe = np.empty_like(μQ)

        for i in range(0, len(μQ)):
            μu[i], μd[i], μs[i], μe[i] = self.solve(μQ[i])
            print(f"μQ = {μQ[i]:.2f}, μu = {μu[i]:.2f}, μd = {μd[i]:.2f}, ", end="")
            print(f"μs = {μs[i]:.2f}, μe = {μe[i]:.2f}")

        mu = np.full_like(μQ, muf)
        md = np.full_like(μQ, mdf)
        ms = np.full_like(μQ, msf)
        return mu, md, ms, μu, μd, μs, μe

class LSMModel(Model):
    def eossolve(self, N):
        Δx = np.linspace(self.mu, 0, N)[:-1] # shave off erronous 0
        Δy = np.empty_like(Δx)
        μQ = np.empty_like(Δx)
        μu = np.empty_like(Δx)
        μd = np.empty_like(Δx)
        μs = np.empty_like(Δx)
        μe = np.empty_like(Δx)

        for i in range(0, len(Δx)):
            mu, md ,ms = self.vacuum_masses()
            guess = (μQ[i-1], Δy[i-1], μe[i-1]) if i > 0 else (mu, ms, 0) # use prev sol
            μQ[i], Δy[i], μu[i], μd[i], μs[i], μe[i] = self.solve(Δx[i], guess)
            print(f"Δx = {Δx[i]:.2f}, Δy = {Δy[i]:.2f}, ", end="")
            print(f"μu = {μu[i]:.2f}, μd = {μd[i]:.2f}, ", end="")
            print(f"μs = {μs[i]:.2f}, μe = {μe[i]:.2f}")

        return Δx, Δx, Δy, μu, μd, μs, μe

    def vacuum_masses(self):
        min = scipy.optimize.minimize(
            lambda ΔxΔy: self.Ω(ΔxΔy[0], ΔxΔy[0], ΔxΔy[1], 0, 0, 0, 0),
            x0=(mu0, ms0), method="Nelder-Mead"
        )
        if min.success:
            Δx0, Δy0 = min.x
        else:
            Δx0, Δy0 = np.nan, np.nan
        return Δx0, Δx0, Δy0

    def vacuum_potential(self, Δx, Δy, write=False):
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig, axl = plt.subplots()
        axr = axl.twinx()
        ΔxΔx, ΔyΔy = np.meshgrid(Δx, Δy)

        Ωf = lambda Δx, Δy: self.Ω(Δx, Δx, Δy, 0, 0, 0, 0) / fπ**4 # in vacuum

        Ω = Ωf(ΔxΔx, ΔyΔy)
        Ω0 = np.max(np.abs(Ω))
        mlab.mesh(ΔxΔx / Δx[-1], ΔyΔy / Δy[-1], Ω / Ω0)
        mlab.mesh(ΔxΔx / Δx[-1], ΔyΔy / Δy[-1], Ω / Ω0, representation="wireframe")
        mlab.axes()

        Δx0, _, Δy0 = self.vacuum_masses()
        if not np.isnan(Δx0) and not np.isnan(Δy0):
            print(f"mσ = {self.mσ} MeV: found minimum (Δx, Δy, Ω/fπ^4) = ", end="")
            print(f"({Δx0:.0f} MeV, {Δy0:.0f} MeV, {Ωf(Δx0, Δy0)})")
            Ωx0 = Ωf(Δx, Δy0)
            Ωy0 = Ωf(Δx0, Δy)
            mlab.plot3d(np.full(Δy.shape, Δx0) / Δx[-1], Δy / Δy[-1], Ωy0 / Ω0)
            mlab.plot3d(Δx / Δx[-1], np.full(Δx.shape, Δy0) / Δy[-1], Ωx0 / Ω0)
        else:
            print(f"mσ = {self.mσ} MeV: no minimum!")

        if write:
            cols  = [ΔxΔx.flatten(), ΔyΔy.flatten(), Ω.flatten()] 
            heads = ["Deltax", "Deltay", "Omega"]
            utils.writecols(
                cols, heads, f"data/{self.name}/potential_vacuum_sigma{mσ}.dat",
                skipevery=len(Δx)
            )

        mlab.show()

class LSM2FlavorModel(LSMModel):
    def __init__(self, mσ=mσ, mπ=mπ, renormalize=True):
        Nf = 2
        m2 = 1/2*(3*mπ**2-mσ**2)
        λ  = 3/fπ**2 * (mσ**2-mπ**2)
        h  = fπ * mπ**2
        g  = mu0 / fπ
        Λ2 = mu0**2 / np.e
        print(f"m2  = {np.sign(m2)}*({np.sqrt(np.abs(m2))} MeV)^2 ")
        print(f"λ   = {λ}")
        print(f"g   = {g}")
        print(f"h   = {np.sign(h)}*({np.abs(h)**(1/3)} MeV)^3 ")
        print(f"Λ   = {np.sqrt(Λ2)} MeV")

        Δ, μu, μd, μe = sp.symbols("Δ μ_u μ_d μ_e", complex=True)
        σ = Δ / g
        Ω0 = 1/2*m2*σ**2 + λ/24*σ**4 - h*σ
        Ωr = Nc*Nf*Δ**4/(16*π**2)*(3/2+sp.log(Λ2/Δ**2)) if renormalize else 0
        Ωu = -Nc/(24*π**2)*((2*μu**2-5*Δ**2)*μu*sp.sqrt(μu**2-Δ**2)+\
             3*Δ**4*sp.asinh(sp.sqrt(μu**2/Δ**2-1)))
        Ωd = -Nc/(24*π**2)*((2*μd**2-5*Δ**2)*μd*sp.sqrt(μd**2-Δ**2)+\
             3*Δ**4*sp.asinh(sp.sqrt(μd**2/Δ**2-1)))
        Ωe = -1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+\
             3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))
        Ω  = Ω0 + Ωr + Ωu + Ωd + Ωe
        dΩ = sp.diff(Ω, Δ)

        Ω  = sp.lambdify((Δ, μu, μd, μe),  Ω, "numpy")
        dΩ = sp.lambdify((Δ, μu, μd, μe), dΩ, "numpy")
        self.Ω  = lambda mu,md,ms,μu,μd,μs,μe: np.real( Ω(mu+0j,μu+0j,μd+0j,μe+0j))
        self.dΩ = lambda mu,md,ms,μu,μd,μs,μe: np.real(dΩ(mu+0j,μu+0j,μd+0j,μe+0j))

        Model.__init__(self, "LSM2F", mσ=mσ, mπ=mπ)

    def solve(self, Δx, guess):
        def system(μQ_Δy_μe):
            μQ, Δy, μe = μQ_Δy_μe # unpack variables
            μu, μd, _ = μelim(μQ, μe)
            μs = 0
            # hack to give Δy = 0
            return (self.dΩ(Δx, Δx, 0, μu, μd, 0, μe), Δy,
                    charge(Δx, Δx, 0, μu, μd, 0, μe))
        sol = scipy.optimize.root(system, guess, method="lm") # lm and krylov works
        assert sol.success, f"{sol.message} (Δx = {Δx})"
        μQ, Δy, μe = sol.x
        μu, μd, _ = μelim(μQ, μe)
        Δy, μs = 0, 0
        return μQ, Δy, μu, μd, μs, μe

class LSM2FlavorConsistentModel(LSM2FlavorModel):
    def __init__(self, mσ=mσ, mπ=mπ):
        Δ, μu, μd, μe = sp.symbols("Δ μ_u μ_d μ_e", complex=True)
        def r(p2): return sp.sqrt(4*mu0**2/p2-1)
        def F(p2): return 2 - 2*r(p2)*sp.atan(1/r(p2))
        def dF(p2): return 4*mu0**2*r(p2)/(p2*(4*mu0**2-p2))*sp.atan(1/r(p2))-1/p2
        Ω  = 3/4*mπ**2*fπ**2*(1-4*mu0**2*Nc/(4*π*fπ)**2*mπ**2*dF(mπ**2)) * (Δ/mu0)**2
        Ω -= mσ**2*fπ**2/4*(1+4*mu0**2*Nc/(4*π*fπ)**2*((1-4*mu0**2/mσ**2)*F(mσ**2)+\
             4*mu0**2/mσ**2-F(mπ**2)-mπ**2*dF(mπ**2))) * (Δ/mu0)**2
        Ω += mσ**2*fπ**2/8*(1-4*mu0**2*Nc/(4*π*fπ)**2*(4*(mu0/mσ)**2*sp.log((Δ/mu0)**2)-\
             (1-4*mu0**2/mσ**2)*F(mσ**2)+F(mπ**2)+mπ**2*dF(mπ**2))) * (Δ/mu0)**4
        Ω -= mπ**2*fπ**2/8*(1-4*mu0**2*Nc/(4*π*fπ)**2*mπ**2*dF(mπ**2)) * (Δ/mu0)**4
        Ω -= mπ**2*fπ**2*(1-4*mu0**2*Nc/(4*π*fπ)**2*mπ**2*dF(mπ**2)) * Δ/mu0
        Ω += 3*Nc/(16*π**2) * Δ**4
        Ω -= Nc/(24*π**2)*((2*μu**2-5*Δ**2)*μu*sp.sqrt(μu**2-Δ**2)+\
             3*Δ**4*sp.asinh(sp.sqrt(μu**2/Δ**2-1)))
        Ω -= Nc/(24*π**2)*((2*μd**2-5*Δ**2)*μd*sp.sqrt(μd**2-Δ**2)+\
             3*Δ**4*sp.asinh(sp.sqrt(μd**2/Δ**2-1)))
        Ω -= 1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+\
             3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))

        dΩ = sp.diff(Ω, Δ) # (numerical differentiation also works)
        dΩ = sp.lambdify((Δ, μu, μd, μe), dΩ, "numpy")
        self.dΩ = lambda mu, md, ms, μu, μd, μs, μe: np.real(dΩ(mu+0j,μu+0j,μd+0j,μe+0j))

        Ω  = sp.lambdify((Δ, μu, μd, μe),  Ω, "numpy")
        self.Ω  = lambda mu, md, ms, μu, μd, μs, μe: np.real( Ω(mu+0j,μu+0j,μd+0j,μe+0j))

        Model.__init__(self, "LSM2FC", mσ=mσ, mπ=mπ)

class LSM3FlavorModel(LSMModel):
    def __init__(self, mσ=mσ, mπ=mπ, mK=mK):
        def system(m2_λ1_λ2):
            m2, λ1, λ2 = m2_λ1_λ2
            m2σσ00 = m2 + λ1/3*(4*np.sqrt(2)*σx0*σy0+7*σx0**2+5*σy0**2) + \
                     λ2*(σx0**2+σy0**2)
            m2σσ88 = m2 - λ1/3*(4*np.sqrt(2)*σx0*σy0-5*σx0**2-7*σy0**2) + \
                     λ2/2*(σx0**2+4*σy0**2)
            m2σσ08 = 2/3*λ1*(np.sqrt(2)*σx0**2-np.sqrt(2)*σy0**2-σx0*σy0) + \
                     λ2/np.sqrt(2)*(σx0**2-2*σy0**2)
            m2ππ11 = m2 + λ1*(σx0**2+σy0**2) + λ2/2*σx0**2
            m2ππ44 = m2 + λ1*(σx0**2+σy0**2) - λ2/2*(np.sqrt(2)*σx0*σy0-σx0**2-2*σy0**2)
            θσ = np.arctan(2*m2σσ08 / (m2σσ88-m2σσ00)) / 2
            m2σ = m2σσ00*np.cos(θσ)**2 + m2σσ88*np.sin(θσ)**2 - m2σσ08*np.sin(2*θσ)
            m2π = m2ππ11
            m2K = m2ππ44
            return (m2σ - mσ**2, m2π - mπ**2, m2K - mK**2)

        sol = scipy.optimize.root(system, (-100, -10, +100), method="hybr")
        m2, λ1, λ2 = sol.x
        g = 2*mu0/σx0
        hx = σx0 * (m2 + λ1*(σx0**2+σy0**2) + λ2/2*σx0**2)
        hy = σy0 * (m2 + λ1*(σx0**2+σy0**2) + λ2*σy0**2)
        Λx = g*σx0/(2*np.sqrt(np.e))
        Λy = g*σy0/(np.sqrt(2*np.e))
        common_renormalization_scale = False
        if common_renormalization_scale:
            Λ = (2*Λx+Λy)/3
            Λx = Λ # set common, averaged renormalization scale
            Λy = Λ
        print(f"m2 = {np.sign(m2)}*({np.sqrt(np.abs(m2))} MeV)^2 ")
        print(f"λ1 = {λ1}")
        print(f"λ2 = {λ2}")
        print(f"g  = {g}")
        print(f"hx = ({hx**(1/3)} MeV)^3")
        print(f"hy = ({hy**(1/3)} MeV)^3")
        print(f"Λx = {Λx} MeV")
        print(f"Λy = {Λy} MeV")
        print(f"mx = {g*σx0/2} MeV")
        print(f"my = {g*σy0/np.sqrt(2)} MeV")

        Δx, Δy, μu, μd, μs, μe = sp.symbols("Δ_x Δ_y μ_u μ_d μ_s μ_e", complex=True)
        σx = 2*Δx/g
        σy = np.sqrt(2)*Δy/g
        Ωb = m2/2*(σx**2+σy**2) + λ1/4*(σx**2+σy**2)**2 + λ2/8*(σx**4+2*σy**4) -\
             hx*σx - hy*σy 
        Ωr = Nc/(16*π**2)*(Δx**4*(3/2+sp.log(Λx**2/Δx**2))+\
             Δx**4*(3/2+sp.log(Λx**2/Δx**2))+Δy**4*(3/2+sp.log(Λy**2/Δy**2)))
        Ωu = -Nc/(24*π**2)*((2*μu**2-5*Δx**2)*μu*sp.sqrt(μu**2-Δx**2)+\
             3*Δx**4*sp.asinh(sp.sqrt(μu**2/Δx**2-1)))
        Ωd = -Nc/(24*π**2)*((2*μd**2-5*Δx**2)*μd*sp.sqrt(μd**2-Δx**2)+\
             3*Δx**4*sp.asinh(sp.sqrt(μd**2/Δx**2-1)))
        Ωs = -Nc/(24*π**2)*((2*μs**2-5*Δy**2)*μs*sp.sqrt(μs**2-Δy**2)+\
             3*Δy**4*sp.asinh(sp.sqrt(μs**2/Δy**2-1)))
        Ωe =  -1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+\
             3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))

        Ω  = Ωb + Ωr + Ωu + Ωd + Ωs + Ωe
        dΩx = sp.diff(Ω, Δx)
        dΩy = sp.diff(Ω, Δy)

        Ω   = sp.lambdify((Δx, Δy, μu, μd, μs, μe), Ω,   "numpy")
        dΩx = sp.lambdify((Δx, Δy, μu, μd, μs, μe), dΩx, "numpy")
        dΩy = sp.lambdify((Δx, Δy, μu, μd, μs, μe), dΩy, "numpy")
        self.Ω   = lambda mu, md, ms, μu, μd, μs, μe: \
                   np.real(  Ω(mu+0j, ms+0j, μu+0j, μd+0j, μs+0j, μe+0j))
        self.dΩx = lambda mu, md, ms, μu, μd, μs, μe: \
                   np.real(dΩx(mu+0j, ms+0j, μu+0j, μd+0j, μs+0j, μe+0j))
        self.dΩy = lambda mu, md, ms, μu, μd, μs, μe: \
                   np.real(dΩy(mu+0j, ms+0j, μu+0j, μd+0j, μs+0j, μe+0j))

        Model.__init__(self, f"LSM3F", mσ=mσ, mπ=mπ, mK=mK)

    def solve(self, Δx, guess):
        def system(μQ_Δy_μe):
            μQ, Δy, μe = μQ_Δy_μe # unpack variables
            μu, μd, μs = μelim(μQ, μe)
            return (self.dΩx(Δx, Δx, Δy, μu, μd, μs, μe),
                    self.dΩy(Δx, Δx, Δy, μu, μd, μs, μe),
                    charge(Δx, Δx, Δy, μu, μd, μs, μe))
        sol = scipy.optimize.root(system, guess, method="lm")
        assert sol.success, f"{sol.message} (Δx = {Δx})"
        μQ, Δy, μe = sol.x
        μu, μd, μs = μelim(μQ, μe)
        return μQ, Δy, μu, μd, μs, μe

class HybridModel(Model):
    def eos(self, N=1000, B=111**4, hybrid=True, plot=False, write=False):
        arr = np.loadtxt("data/APR/eos.dat")
        #mn = 939.56542052 # MeV / c^2
        mn = 900
        nB = arr[:,0]
        P_over_nB = arr[:,1]
        μB_over_mn_minus_one = arr[:,3]
        ϵ_over_nBmn_minus_one = arr[:,6]
        P = P_over_nB * nB
        ϵ = (ϵ_over_nBmn_minus_one + 1) * (nB*mn)

        nB1 = nB
        μB1 = (μB_over_mn_minus_one + 1) * mn
        P1  = P * 1e-3 * (GeV/fm**3) / ϵ0 # MeV/fm^3 -> GeV/fm^3 -> SI -> TOV-dimless
        ϵ1  = ϵ * 1e-3 * (GeV/fm**3) / ϵ0 # MeV/fm^3 -> GeV/fm^3 -> SI -> TOV-dimless
        
        ϵ2int, nu2int, nd2int, ns2int, _, μQ2int = self.quarkmodel(
            mσ=self.mσ).eos(N=N-len(P1), B=B
        )
        nB2 = (nu2int(P1)+nd2int(P1)+ns2int(P1)) / 3
        μB2 = μQ2int(P1) * 3

        if plot:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(μB1, P1, "-b.")
            ax1.plot(μB2, P1, "-r.")
            ax2.plot(μB1, nB1/nsat, "-b.")
            ax2.plot(μB1, nB2/nsat, "-r.")
            #plt.scatter(nB0, P0)
            plt.show()

        # find intersecting nB (from top)
        P1i = scipy.interpolate.interp1d(μB1, P1)
        P2i = scipy.interpolate.interp1d(μB2, P1)
        sol = scipy.optimize.root_scalar(
            lambda μB: P2i(μB)-P1i(μB), method="brentq", bracket=(1200, 2000)
        )
        assert sol.converged
        μB0 = sol.root
        if not hybrid:
            μB0 = 2700 # will use only hadronic EOS
        P0 = P1i(μB0)
        print(f"μB0 = {μB0}")
        print(f"P0  = {P0} = 10^({np.log10(P0*ϵ0)}) Pa")
        print(f"Δϵ  = {ϵ2int(P0)-np.interp(P0, P1, ϵ1) * ϵ0 / (GeV/fm**3)}")

        # should it be stable? see ref:hybrid_star_stability_criterion, equation 15
        lhs = ϵ2int(P0)-np.interp(P0, P1, ϵ1)
        rhs = np.interp(P0,P1,ϵ1)/2+3/2*P0
        print(f"Should be stable? {lhs} < {rhs} ? {lhs < rhs}")

        # compute pressure for larger values to increase interpolation range
        P2 = np.linspace(P0, 1e-1, N-len(P1))
        ϵ2 = ϵ2int(P2)
        P  = np.concatenate((P1[P1<P0], [P0], P2))
        ϵ  = np.concatenate((ϵ1[P1<P0], [np.interp(P0, P1, ϵ1)], ϵ2))

        if plot:
            plt.plot(P, ϵ, "-k.", linewidth=4)
            plt.plot(P1, ϵ1, "-b.")
            plt.plot(P, ϵ2int(P), "-r.")
            plt.show()

        # hack: exploit nu=nd=ns=nB for density interpolation
        nB1 = nB1 # already have from data set
        nB2 = (nu2int(P2)+nd2int(P2)+ns2int(P2)) / 3 # compute with P2 instead of P1
        nB = np.concatenate((nB1[P1<P0], [np.interp(P0, P1, nB1)], nB2))
        nu = np.concatenate((nB1[P1<P0], [np.interp(P0, P1, nB1)], nu2int(P2)))
        nd = np.concatenate((nB1[P1<P0], [np.interp(P0, P1, nB1)], nd2int(P2)))
        ns = np.concatenate((nB1[P1<P0], [np.interp(P0, P1, nB1)], ns2int(P2)))

        μB1 = μB1 # already have from data set
        μB2 = μQ2int(P2) * 3
        μB = np.concatenate((μB1[P1<P0], [np.interp(P0, P1, μB1)], μB2))

        ϵ = np.concatenate(([0], ϵ))
        P = np.concatenate(([-10], P)) # force ϵ(P<Pmin)=0 (avoid interpolation errors)
        nB = np.concatenate(([0], nB))
        nu = np.concatenate(([0], nu))
        nd = np.concatenate(([0], nd))
        ns = np.concatenate(([0], ns))
        μB = np.concatenate(([0], μB))
        ϵint = scipy.interpolate.interp1d(P, ϵ); ϵint.__name__ = self.name
        nBint = scipy.interpolate.interp1d(P, nB)
        nuint = scipy.interpolate.interp1d(P, nu)
        ndint = scipy.interpolate.interp1d(P, nd)
        nsint = scipy.interpolate.interp1d(P, ns)
        μBint = scipy.interpolate.interp1d(P, μB)

        # ignore electrons
        zerofunc = lambda x: 0*x # works for scalars and arrays

        if write:
            nB2 = (nu2int(P)+nd2int(P)+ns2int(P)) / 3 # compute with P instead
            μB2 = μQ2int(P) * 3 # compute with P instead
            ϵ1 = ϵ1 * ϵ0 / (GeV/fm**3)
            ϵ2 = np.concatenate(([0], ϵ2int(P[1:]))) * ϵ0 / (GeV/fm**3) # skip -10
            ϵ  = ϵ * ϵ0 / (GeV/fm**3)
            P  = P * ϵ0 / (GeV/fm**3)
            P1 = P1* ϵ0 / (GeV/fm**3)
            cols  = [nB, nB1, nB2, μB, μB1, μB2, P, P1, ϵ, ϵ1, ϵ2]
            heads = ["nB", "nB1", "nB2", "muB", "muB1", "muB2", "P", "P1",
                     "epsilon", "epsilon1", "epsilon2"]
            outfile = f"data/{self.name}/eos_sigma_{self.mσ}.dat"
            utils.writecols(cols, heads, outfile)

        μQint = lambda P: μBint(P) / 3
        return ϵint, nuint, ndint, nsint, zerofunc, μQint

class Hybrid2FlavorModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM2F_APR"
        self.mσ = mσ
        self.quarkmodel = LSM2FlavorModel

class Hybrid3FlavorModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM3F_APR"
        self.mσ = mσ
        self.quarkmodel = LSM3FlavorModel

class Hybrid2FlavorConsistentModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM2FC_APR"
        self.mσ = mσ
        self.quarkmodel = LSM2FlavorConsistentModel

if __name__ == "__main__":
    # plot 3D potential for 2-flavor model with μu=μd
    """
    mσ = 700
    model = LSM2FlavorModel(mσ=mσ)
    Δ = np.linspace(-1000, +1000, 100)
    μQ = np.linspace(0, 500, 50)
    Ω = np.array([model.Ω(Δ, Δ, 0, μQ, μQ, 0, 0) for μQ in μQ])
    Δ0 = np.empty_like(μQ)
    Ω0 = np.empty_like(μQ)
    for i in range(0, len(μQ)):
        μQ0 = μQ[i]
        def Ω2(Δ): return model.Ω(Δ, Δ, 0, μQ0, μQ0, 0, 0)
        sol = scipy.optimize.minimize_scalar(Ω2, bounds=(0, 350), method="bounded")
        assert sol.success, f"{sol.message} (μ = {μQ0})"
        Δ0[i] = sol.x
        Ω0[i] = Ω2(Δ0[i])
    plt.xlabel(r"$\Delta$")
    plt.ylabel(r"$\Omega$")
    plt.plot(Δ, Ω.T / 100**4, "-k")
    plt.plot(Δ0, Ω0 / 100**4, "-r.")
    plt.show()
    Δc, μQc, Ωc = [], [], []
    for i in range(0, len(Δ)):
        for j in range(0, len(μQ)):
            μQc.append(μQ[j])
            Δc.append(Δ[i])
            Ωc.append(Ω[j,i])
    cols = [μQc, Δc, list(np.array(Ωc)/100**4), μQc, list(Δ0), list(Ω0/100**4)]
    heads = ["mu", "Delta", "Omega", "mu0", "Delta0", "Omega0"]
    utils.writecols(
        cols, heads, f"data/{model.name}/potential_noisospin_sigma_{mσ}.dat",
        skipevery=len(μQ)
    )
    exit()
    """

    # MIT bag models (2 and 3 flavors)
    """
    models = [MIT2FlavorModel, MIT3FlavorModel]
    for model in models:
        model = model()
        model.eos(plot=False, write=True)
        for B14 in (145, 150, 155):
            model.stars(B14, (1e-7, 1e-2), write=True)
    exit()
    """

    # vacuum potentials
    """
    Δ = np.linspace(-600, +600, 300)
    for mσ in [500, 600, 700, 800]:
        LSM2FlavorModel(mσ=mσ).vacuum_potential(Δ, np.array([ms0]), write=True)
    for mσ in [400, 500, 600, 700, 800]:
        LSM2FlavorConsistentModel(mσ=mσ).vacuum_potential(Δ, np.array([ms0]), write=True)
    Δ = np.linspace(-1000, +1000, 50)
    for mσ in [500, 600, 700, 800]:
        LSM3FlavorModel(mσ=mσ).vacuum_potential(Δ, Δ, write=True)
    exit()
    """

    # 2-flavor quark-meson model
    #LSM2FlavorModel(mσ=600).eos(write=True)
    #LSM2FlavorModel(mσ=700).eos(write=True)
    #LSM2FlavorModel(mσ=800).eos(write=True)
    #LSM2FlavorModel(mσ=600).stars(111, (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=600).stars(131, (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=600).stars(151, (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=700).stars(68,  (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=700).stars(88,  (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=700).stars(108, (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=800).stars(27,  (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=800).stars(47,  (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=800).stars(67,  (1e-7, 1e-2), write=True)
    #LSM2FlavorModel(mσ=800).star(0.0012500875, 27, write=True)
    #exit()

    # 2-flavor consistent quark-meson model
    #LSM2FlavorConsistentModel(mσ=400).eos(plot=False, write=True)
    #LSM2FlavorConsistentModel(mσ=500).eos(plot=False, write=True)
    #LSM2FlavorConsistentModel(mσ=600).eos(plot=False, write=True)
    #LSM2FlavorConsistentModel(mσ=400).stars(107, (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=400).stars(127, (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=400).stars(147, (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=500).stars(84,  (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=500).stars(104, (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=500).stars(124, (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=600).stars(27,  (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=600).stars(47,  (1e-7, 1e-2), write=True)
    #LSM2FlavorConsistentModel(mσ=600).stars(67,  (1e-7, 1e-2), write=True)
    #exit()

    # 3-flavor quark-meson model
    #LSM3FlavorModel(mσ=600).eos(plot=False, write=True)
    #LSM3FlavorModel(mσ=700).eos(plot=False, write=True)
    #LSM3FlavorModel(mσ=800).eos(plot=False, write=True)
    #LSM3FlavorModel(mσ=600).stars(111, (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=600).stars(131, (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=600).stars(151, (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=700).stars(68,  (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=700).stars(88,  (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=700).stars(108, (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=800).stars(27,  (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=800).stars(47,  (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=800).stars(67,  (1e-7, 1e-2), write=True)
    #LSM3FlavorModel(mσ=800).star(0.000937590625, 27, write=True)
    #exit()

    # strange quark stars (need B^(1/4) ≳ 145 MeV in all models)
    # TODO: finish considering this
    #LSM3FlavorModel(mσ=800).eos(B=0**4, plot=True)
    #LSM3FlavorModel(mσ=600).star(1e-3, 145, plot=True)
    #for mσ in (600, 700, 800):
        #LSM3FlavorModel(mσ=mσ).stars(145, (1e-7, 1e-2), write=True)
        #LSM3FlavorModel(mσ=mσ).stars(165, (1e-7, 1e-2), write=True)
        #LSM3FlavorModel(mσ=mσ).stars(185, (1e-7, 1e-2), write=True)

    # hybrid model (2-flavor quark-meson model + APR hadronic EOS)
    #Hybrid2FlavorModel(mσ=600).eos(B=111**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=700).eos(B=68**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=800).eos(B=27**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=600).stars(111, (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid2FlavorModel(mσ=700).stars(68,  (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid2FlavorModel(mσ=800).stars(27,  (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid2FlavorModel(mσ=600).star(0.001180703125, 111, plot=True, write=True)
    #exit()

    # hybrid model (3-flavor quark-meson model + APR hadronic EOS)
    #Hybrid3FlavorModel(mσ=600).eos(B=111**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=700).eos(B=68**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=800).eos(B=27**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=600).stars(111, (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid3FlavorModel(mσ=700).stars(68,  (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid3FlavorModel(mσ=800).stars(27,  (1e-5, 1e-2), write=True) # use tolD=0.01
    #Hybrid3FlavorModel(mσ=600).star(0.0008160778808593749, 111, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=800).star(0.0012587499999999999, 27, plot=True, write=True)
    #exit()

    # consistent hybrid model (2-flavor consistent quark-meson omdel + APR hadronic EOS)
    #Hybrid2FlavorConsistentModel(mσ=400).eos(B=107**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=500).eos(B=84**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=600).eos(B=27**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=400).stars(107, (1e-5, 1e-2), write=True, plot=True) # use tolD=0.01
    #Hybrid2FlavorConsistentModel(mσ=500).stars(84,  (1e-5, 1e-2), write=True, plot=True) # use tolD=0.01
    #Hybrid2FlavorConsistentModel(mσ=600).stars(27,  (1e-5, 1e-2), write=True, plot=True) # use tolD=0.01
    #Hybrid2FlavorConsistentModel(mσ=400).star(0.001180703125, 107, plot=True, write=True)

    #exit()
