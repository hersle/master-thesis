#!/usr/bin/python3

from constants import π, ħ, c, ϵ0, MeV, GeV, fm
from tov import massradiusplot, soltov
import utils

import numpy as np
import sympy as sp
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

Nc = 3
me = 0.5

tovopts = {"tolD": 0.01, "maxdr": 1e-2, "nmodes": 0}

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
    def __init__(self, name, mσ=0, mπ=0, mK=0):
        self.name = name
        self.mu, self.md, self.ms = self.vacuum_masses()
        print(f"Meson masses: mσ = {mσ:.1f} MeV, mπ = {mπ:.1f}, mK = {mK:.1f}")
        print(f"Quark masses: mu = md = {self.mu:.1f} MeV, ms = {self.ms:.1f} MeV")
        self.mσ = mσ
        self.mπ = mπ
        self.mK = mK

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
