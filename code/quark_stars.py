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
mu = 300
md = mu
ms = 429 # only for guess
mσ = 800
mπ = 138
mK = 496
me = 0.5

σx0 = fπ
σy0 = np.sqrt(2)*fK-fπ/np.sqrt(2)

tovopts = {"tolD": 0.500, "maxdr": 1e-2, "nmodes": 0}
tovμQ = np.concatenate([np.linspace(0, 700, 200)[1:], np.linspace(700, 5000, 100)])

def charge(Δx, Δy, μu, μd, μs, μe):
    nu = Nc/(3*π**2) * np.real((μu**2-Δx**2+0j)**(3/2))
    nd = Nc/(3*π**2) * np.real((μd**2-Δx**2+0j)**(3/2))
    ns = Nc/(3*π**2) * np.real((μs**2-Δy**2+0j)**(3/2))
    ne =  1/(3*π**2) * np.real((μe**2-me**2+0j)**(3/2))
    return +2/3*nu - 1/3*nd - 1/3*ns - 1*ne

# solve μQ=(μu+μd)/2, μd=μu+μe, μs=μd for (μu,μd,μs)
def μelim(μQ, μe):
    μu = μQ - μe/2
    μd = μu + μe
    μs = μd
    return μu, μd, μs

class Model:
    def __init__(self, name, mu=mu, md=md, ms=ms, mσ=mσ, mπ=mπ, mK=mK):
        self.name = name
        self.mu = mu
        self.md = md
        self.ms = ms
        self.mσ = mσ
        self.mπ = mπ
        self.mK = mK

    def eos(self, Δx, maxwell=True, B=None, nint=False, name="ϵ", plot=False, write=False):
        Δy = np.empty_like(Δx)
        μQ = np.empty_like(Δx)
        μu = np.empty_like(Δx)
        μd = np.empty_like(Δx)
        μs = np.empty_like(Δx)
        μe = np.empty_like(Δx)
        Ω  = np.empty_like(Δx)

        for i in range(0, len(Δx)):
            guess = (μQ[i-1], Δy[i-1], μe[i-1]) if i > 0 else (mu, ms, 0) # use previous solution
            μQ[i], Δy[i], μu[i], μd[i], μs[i], μe[i] = self.solve(Δx[i], guess)
            Ω[i] = self.Ω(Δx[i], Δy[i], μu[i], μd[i], μs[i], μe[i])
            print(f"Δx = {Δx[i]:.2f}, Δy = {Δy[i]:.2f}, ", end="")
            print(f"μQ = {μQ[i]:.2f}, μu = {μu[i]:.2f}, μd = {μd[i]:.2f}, μs = {μs[i]:.2f}, μe = {μe[i]:.2f}")

        P, P0 = -Ω, -Ω[0] + B
        P = P - P0

        nu = Nc/(3*π**2) * np.real((μu**2-Δx**2+0j)**(3/2))
        nd = Nc/(3*π**2) * np.real((μd**2-Δx**2+0j)**(3/2))
        ns = Nc/(3*π**2) * np.real((μs**2-Δy**2+0j)**(3/2))
        ne =  1/(3*π**2) * np.real((μe**2-me**2+0j)**(3/2))
        ϵ  = -P + μu*nu + μd*nd + μs*ns + μe*ne

        P1 = P[0]
        i2 = np.argmax(np.gradient(P) < 0) - 1 # last index of increasing pressure
        have_phase_transition = (i2 != -1)
        print("Phase transition?", have_phase_transition)
        if have_phase_transition:
            P2 = P[i2]
            i3 = i2 + np.argmax(np.gradient(P[i2:]) > 0) - 1
            P3 = P[i3]

            # debug Maxwell construction
            #plt.plot(P[i2+1], ϵ[i2+1], marker=".", color="red")
            #plt.plot(P[i2:i3+1], ϵ[i2:i3+1], marker=".", color="green")
            #plt.plot(P[i3:], ϵ[i3:], marker=".", color="blue")
            #plt.show()

            def dG(P0):
                j1 = np.argmax(P[:i2+1] >= P0) # first index on 1-curve with greater pressure
                j2 = i3 + np.argmax(P[i3:] >= P0) # first index on 2-curve with greater pressure
                ϵ1 = np.interp(P0, P[:i2+1], ϵ[:i2+1])
                ϵ2 = np.interp(P0, P[i3:], ϵ[i3:])
                P12 = np.concatenate([[P0], P[j1:j2], [P0]])
                ϵ12 = np.concatenate([[ϵ1], ϵ[j1:j2], [ϵ2]])
                return np.trapz(1/ϵ12, P12)

            sol = scipy.optimize.root_scalar(dG, bracket=(P[1], P[i2]), method="brentq") # pray that P[1] works, since P[0] = 0 gives 0-div error
            assert sol.converged
            P0 = sol.root

            #fig, (ax1, ax2) = plt.subplots(1, 2)
            #ax1.plot(1/ϵ, P, color="gray")
            #ax1.set_ylim(1.1*np.min(P), -1.1*np.min(P))
            #ax2.plot(P, ϵ, color="gray")

            j1 = np.argmax(P[:i2+1] >= P0) # first index on 1-curve with greater pressure
            j2 = i3 + np.argmax(P[i3:] >= P0) # first index on 2-curve with greater pressure
            ϵ1 = np.interp(P0, P[:i2+1], ϵ[:i2+1])
            ϵ2 = np.interp(P0, P[i3:], ϵ[i3:])

            def fixarray(a):
                a1 = np.interp(P0, P[:i2+1], a[:i2+1])
                a2 = np.interp(P0, P[i3:], a[i3:])
                return np.concatenate((a[:j1], [a1, np.nan, a2], a[j2:]))

            # also keep P, ϵ before Maxwell construction (for comparison)
            Porg, ϵorg = P, ϵ

            # fix array by only modifying EOS, but fill out with points in phase transition
            ϵ1 = np.interp(P0, P[:i2+1], ϵ[:i2+1])
            ϵ2 = np.interp(P0, P[i3:], ϵ[i3:])
            Ntarget = len(Δx)
            Nnow = len(ϵ[:j1]) + len(ϵ[j2:])
            Nadd = Ntarget - Nnow
            ϵ = np.concatenate((ϵ[:j1], np.linspace(ϵ1, ϵ2, Nadd), ϵ[j2:]))
            P = np.concatenate((P[:j1], np.linspace(P0, P0, Nadd), P[j2:]))

        # print bag constant bound (upper or lower, depending on circumstances)
        # TODO: is this correct? particularly with phase transition, where B is added BEFORE this is done?
        nB = 1/3*(nu+nd+ns)
        ϵB = 0 + μu*nu + μd*nd + μs*ns + μe*ne
        Bmin = np.interp(930, ϵB/nB, P+B)
        print(f"bag constant bound: B^(1/4) = {Bmin**(1/4)} MeV")

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

        # interpolate dimensionless EOS
        Pdimless = P / ϵ0 # now in TOV-dimensionless units
        ϵdimless = ϵ / ϵ0 # now in TOV-dimensionless units
        print(f"interpolation range: {Pdimless[0]} < P < {Pdimless[-1]}")
        ϵint = scipy.interpolate.interp1d(Pdimless, ϵdimless)
        ϵint.__name__ = name

        # convert interesting quantities to appropriate units
        P0 *= fm**3 / GeV # now in units GeV/fm^3
        P  *= fm**3 / GeV # now in units GeV/fm^3
        ϵ  *= fm**3 / GeV # now in units GeV/fm^3
        Porg *= fm**3 / GeV # now in units GeV/fm^3
        ϵorg *= fm**3 / GeV # now in units GeV/fm^3

        print(f"P0 = {P0}")

        if write:
            cols  = [μQ, Δx, Δy, μu, μd, μs, μe, nu, nd, ns, ne, ϵ, P]
            heads = ["muQ", "deltax", "deltay", "muu", "mud", "mus", "mue", "nu", "nd", "ns", "ne", "epsilon", "P"]
            if have_phase_transition:
                cols += [ϵorg, Porg]
                heads += ["epsilonorg", "Porg"]
            outfile = f"data/{self.name}/eos.dat"
            utils.writecols(cols, heads, outfile)

        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

            ax1.set_xlabel(r"$\mu_Q$")
            ax1.plot(μQ, Δx, ".-", color="orange", label=r"$\Delta_x$")
            ax1.plot(μQ, Δy, ".-", color="yellow", label=r"$\Delta_y$")
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
            if have_phase_transition:
                ax3.plot(Porg, ϵorg, ".-", color="gray") # compare
            ax3.plot(P, ϵ, ".-", color="black")

            plt.show()

        nuint = scipy.interpolate.interp1d(Pdimless, nu)
        ndint = scipy.interpolate.interp1d(Pdimless, nd)
        nsint = scipy.interpolate.interp1d(Pdimless, ns)
        neint = scipy.interpolate.interp1d(Pdimless, ne)
        return ϵint, nuint, ndint, nsint, neint

    def star(self, Pc, B14):
        ϵ, nu, nd, ns, ne = self.eos(B=B14**4)
        rs, ms, Ps, αs, ϵs = soltov(ϵ, Pc, maxdr=tovopts["maxdr"])
        nus, nds, nss, nes = nu(Ps), nd(Ps), ns(Ps), ne(Ps)

        heads = ["r", "P", "epsilon", "nu", "nd", "ns", "ne"]
        cols = [list(rs), list(Ps), list(ϵs), list(nus), list(nds), list(nss), list(nes)]
        outfile = f"data/{self.name}/star_B14_{B14}_Pc_{Pc:.7f}.dat"
        utils.writecols(cols, heads, outfile)

    def stars(self, B14s, P1P2, plot=False, write=False):
        Δ = np.linspace(300, 0, 700)[:-1]
        for B14 in B14s:
            outfile = f"data/{self.name}/stars_sigma_{self.mσ}_B14_{B14}.dat" if write else ""
            print(f"B = {B14}^4, outfile = {outfile}")
            ϵ, _, _, _, _ = self.eos(Δ, B=B14**4, plot=False)
            massradiusplot(ϵ, P1P2, **tovopts, visual=plot, outfile=outfile)

    def vacuum_potential(self, Δx, Δy, write=False):
        #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig, axl = plt.subplots()
        axr = axl.twinx()
        ΔxΔx, ΔyΔy = np.meshgrid(Δx, Δy)

        Ωf = lambda Δx, Δy: self.Ω(Δx, Δy, 0, 0, 0, 0) / fπ**4 # in vacuum

        Ω = Ωf(ΔxΔx, ΔyΔy)
        Ω0 = np.max(np.abs(Ω))
        mlab.mesh(ΔxΔx / Δx[-1], ΔyΔy / Δy[-1], Ω / Ω0)
        mlab.mesh(ΔxΔx / Δx[-1], ΔyΔy / Δy[-1], Ω / Ω0, representation="wireframe")
        mlab.axes()

        min = scipy.optimize.minimize(lambda ΔxΔy: Ωf(ΔxΔy[0], ΔxΔy[1]), x0=(mu, ms), method="Nelder-Mead")
        if min.success:
            Δx0, Δy0 = min.x
            print(f"mσ = {mσ} MeV: found minimum (Δx, Δy, Ω/fπ^4) = ({Δx0:.0f} MeV, {Δy0:.0f} MeV, {Ωf(Δx0, Δy0)})")
            Ωx0 = Ωf(Δx, Δy0)
            Ωy0 = Ωf(Δx0, Δy)
            mlab.plot3d(np.full(Δy.shape, Δx0) / Δx[-1], Δy / Δy[-1], Ωy0 / Ω0)
            mlab.plot3d(Δx / Δx[-1], np.full(Δx.shape, Δy0) / Δy[-1], Ωx0 / Ω0)
        else:
            print(f"mσ = {mσ} MeV: no minimum!")
        # TODO: minimum moves in 3-flavor case due to one renormalization scale Λ?

        """
        Δx0, Δy0 = scipy.optimize.minimize(lambda ΔxΔy: Ωf(ΔxΔy[0], ΔxΔy[1]), x0=(mu, ms), method="Nelder-Mead").x
        axl.plot(Δx, Ωf(Δx, Δy0), color="red")
        axl.scatter(Δx0, Ωf(Δx0, Δy0), color="red")
        axr.plot(Δy, Ωf(Δx0, Δy), color="blue")
        axr.scatter(Δy0, Ωf(Δx0, Δy0), color="blue")
        """

        if write:
            cols  = [ΔxΔx.flatten(), ΔyΔy.flatten(), Ω.flatten()] 
            heads = ["Deltax", "Deltay", "Omega"]
            utils.writecols(cols, heads, f"data/{self.name}/potential_vacuum_sigma{mσ}.dat", skipevery=len(Δx))

        mlab.show()
        # plt.show()

class Bag2Flavor(Model):
    def __init__(self):
        Model.__init__(self, "MIT2F")
        self.Ω = lambda Δ, Δy, μu, μd, μs, μe: -Nc/(12*π**2)*μu**4 - Nc/(12*π**2)*μd**4 - 1/(12*π**2)*μe**4

    def solve(self, μQ, guess):
        def system(μe):
            μu, μd, _ = μelim(μQ, μe)
            return charge(0, 0, μu, μd, 0, μe) # hack to give Δy = 0
        sol = scipy.optimize.root_scalar(system, bracket=(0,1e5), method="bisect") # lm and krylov workb
        assert sol.converged, f"{sol.message} (μQ = {μQ})"
        μe = sol.root
        μu, μd, _ = μelim(μQ, μe)
        Δx, Δy, μs = 0, 0, 0
        return Δx, Δy, μu, μd, μs, μe

# has same EOS (ϵ = 3P + 4B) as MIT2F
class Bag3Flavor(Model):
    def __init__(self):
        Model.__init__(self, "MIT3F")
        self.Ω = lambda Δ, Δy, μu, μd, μs, μe: -Nc/(12*π**2)*μu**4 - Nc/(12*π**2)*μd**4 - Nc/(12*π**2)*μs**4 - 1/(12*π**2)*μe**4

    def solve(self, μQ, guess):
        def system(μe):
            μu, μd, μs = μelim(μQ, μe)
            return charge(0, 0, μu, μd, μs, μe) # hack to give Δy = 0
        sol = scipy.optimize.root_scalar(system, bracket=(0,1e5), method="bisect") # lm and krylov workb
        assert sol.converged, f"{sol.message} (μQ = {μQ})"
        μe = sol.root
        μu, μd, μs = μelim(μQ, μe)
        Δx, Δy = 0, 0
        return Δx, Δy, μu, μd, μs, μe

class LSM2Flavor(Model):
    def __init__(self, mu=mu, md=md, mσ=mσ, mπ=mπ, renormalize=True):
        Model.__init__(self, "LSM2F", mu=mu, md=md, mσ=mσ, mπ=mπ)

        Nf = 2
        m2 = 1/2*(3*mπ**2-mσ**2)
        λ  = 3/fπ**2 * (mσ**2-mπ**2)
        h  = fπ * mπ**2
        g  = mu / fπ
        Λ2 = mu**2 / np.e
        print(f"m2 = {np.sign(m2)}*({np.sqrt(np.abs(m2))} MeV)^2 ")
        print(f"λ  = {λ}")
        print(f"g  = {g}")
        print(f"h  = {np.sign(h)}*({np.abs(h)**(1/3)} MeV)^3 ")

        Δ, μu, μd, μe = sp.symbols("Δ μ_u μ_d μ_e", complex=True)
        σ = Δ / g
        Ω0 = 1/2*m2*σ**2 + λ/24*σ**4 - h*σ
        Ωr = Nc*Nf*Δ**4/(16*π**2)*(3/2+sp.log(Λ2/Δ**2)) if renormalize else 0
        Ωu = -Nc/(24*π**2)*((2*μu**2-5*Δ**2)*μu*sp.sqrt(μu**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μu**2/Δ**2-1)))
        Ωd = -Nc/(24*π**2)*((2*μd**2-5*Δ**2)*μd*sp.sqrt(μd**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μd**2/Δ**2-1)))
        Ωe =  -1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))
        Ω  = Ω0 + Ωr + Ωu + Ωd + Ωe
        dΩ = sp.diff(Ω, Δ)

        Ω  = sp.lambdify((Δ, μu, μd, μe),  Ω, "numpy")
        dΩ = sp.lambdify((Δ, μu, μd, μe), dΩ, "numpy")
        self.Ω  = lambda Δ, Δy, μu, μd, μs, μe: np.real( Ω(Δ+0j, μu+0j, μd+0j, μe+0j))
        self.dΩ = lambda Δ, Δy, μu, μd, μs, μe: np.real(dΩ(Δ+0j, μu+0j, μd+0j, μe+0j))

    def solve(self, Δx, guess):
        # TODO: handle phase transition
        # if μQ > 313:
            # guess = (150, guess[1], guess[2])
        def system(μQ_Δy_μe):
            μQ, Δy, μe = μQ_Δy_μe # unpack variables
            μu, μd, _ = μelim(μQ, μe)
            μs = 0
            return (self.dΩ(Δx, 0, μu, μd, 0, μe), Δy, charge(Δx, 0, μu, μd, 0, μe)) # hack to give Δy = 0
        sol = scipy.optimize.root(system, guess, method="lm") # lm and krylov works
        assert sol.success, f"{sol.message} (Δx = {Δx})"
        μQ, Δy, μe = sol.x
        μu, μd, _ = μelim(μQ, μe)
        Δy, μs = 0, 0
        return μQ, Δy, μu, μd, μs, μe

class LSM2FlavorConsistent(LSM2Flavor):
    def __init__(self, mu=mu, md=md, mσ=mσ, mπ=mπ):
        Model.__init__(self, "LSM2FC_sigma600", mu=mu, md=md, mσ=mσ, mπ=mπ)

        Δ, μu, μd, μe = sp.symbols("Δ μ_u μ_d μ_e", complex=True)
        def r(p2): return sp.sqrt(4*mu**2/p2-1)
        def F(p2): return 2 - 2*r(p2)*sp.atan(1/r(p2))
        def dF(p2): return 4*mu**2*r(p2)/(p2*(4*mu**2-p2))*sp.atan(1/r(p2))-1/p2
        Ω  = 3/4*mπ**2*fπ**2*(1-4*mu**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ**2 / mu**2
        Ω -= mσ**2*fπ**2/4*(1+4*mu**2*Nc/(16*π**2*fπ**2)*((1-4*mu**2/mσ**2)*F(mσ**2)+4*mu**2/mσ**2-F(mπ**2)-mπ**2*dF(mπ**2))) * Δ**2/mu**2
        Ω += mσ**2*fπ**2/8*(1-4*mu**2*Nc/(16*π**2*fπ**2)*(4*mu**2/mσ**2*sp.log(Δ**2/mu**2)-(1-4*mu**2/mσ**2)*F(mσ**2)+F(mπ**2)+mπ**2*dF(mπ**2)))* Δ**4 / mu**4
        Ω -= mπ**2*fπ**2/8*(1-4*mu**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ**4/mu**4
        Ω -= mπ**2*fπ**2*(1-4*mu**2*Nc/(16*π**2*fπ**2)*mπ**2*dF(mπ**2)) * Δ/mu
        Ω += 3*Nc/(16*π**2) * Δ**4
        Ω -= Nc/(24*π**2)*((2*μu**2-5*Δ**2)*μu*sp.sqrt(μu**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μu**2/Δ**2-1)))
        Ω -= Nc/(24*π**2)*((2*μd**2-5*Δ**2)*μd*sp.sqrt(μd**2-Δ**2)+3*Δ**4*sp.asinh(sp.sqrt(μd**2/Δ**2-1)))
        Ω -= 1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))

        dΩ = sp.diff(Ω, Δ)
        dΩ = sp.lambdify((Δ, μu, μd, μe), dΩ, "numpy")
        self.dΩ = lambda Δ, Δy, μu, μd, μs, μe: np.real(dΩ(Δ+0j, μu+0j, μd+0j, μe+0j))

        Ω  = sp.lambdify((Δ, μu, μd, μe),  Ω, "numpy")
        self.Ω  = lambda Δ, Δy, μu, μd, μs, μe: np.real( Ω(Δ+0j, μu+0j, μd+0j, μe+0j))

        """
        # TODO: numerical diff also works well
        eps = 0.1
        self.dΩ = lambda Δ, Δy, μu, μd, μs, μe: (self.Ω(Δ+eps/2,Δy,μu,μd,μs,μe)-self.Ω(Δ-eps/2,Δy,μu,μd,μs,μe))/eps
        """

class LSM3Flavor(Model):
    def __init__(self, mu=mu, md=md, ms=ms, mσ=mσ, mπ=mπ, mK=mK):
        Model.__init__(self, f"LSM3F", mu=mu, md=md, ms=ms, mσ=mσ, mπ=mπ, mK=mK)
        def system(m2_λ1_λ2):
            m2, λ1, λ2 = m2_λ1_λ2
            m2σσ00 = m2 + λ1/3*(4*np.sqrt(2)*σx0*σy0+7*σx0**2+5*σy0**2) + λ2*(σx0**2+σy0**2)
            m2σσ88 = m2 - λ1/3*(4*np.sqrt(2)*σx0*σy0-5*σx0**2-7*σy0**2) + λ2/2*(σx0**2+4*σy0**2)
            m2σσ08 = 2/3*λ1*(np.sqrt(2)*σx0**2-np.sqrt(2)*σy0**2-σx0*σy0) + λ2/np.sqrt(2)*(σx0**2-2*σy0**2)
            m2ππ11 = m2 + λ1*(σx0**2+σy0**2) + λ2/2*σx0**2
            m2ππ44 = m2 + λ1*(σx0**2+σy0**2) - λ2/2*(np.sqrt(2)*σx0*σy0-σx0**2-2*σy0**2)
            θσ = np.arctan(2*m2σσ08 / (m2σσ88-m2σσ00)) / 2
            m2σ = m2σσ00*np.cos(θσ)**2 + m2σσ88*np.sin(θσ)**2 - m2σσ08*np.sin(2*θσ)
            m2π = m2ππ11
            m2K = m2ππ44
            return (m2σ - mσ**2, m2π - mπ**2, m2K - mK**2)

        sol = scipy.optimize.root(system, (-100, -10, +100), method="hybr")
        m2, λ1, λ2 = sol.x
        g = 2*mu/σx0
        hx = σx0 * (m2 + λ1*(σx0**2+σy0**2) + λ2/2*σx0**2)
        hy = σy0 * (m2 + λ1*(σx0**2+σy0**2) + λ2*σy0**2)
        Λx = g*σx0/(2*np.sqrt(np.e))
        Λy = g*σx0/(np.sqrt(2*np.e))
        Λ = (2*Λx+Λy)/3
        print(f"m2 = {np.sign(m2)}*({np.sqrt(np.abs(m2))} MeV)^2 ")
        print(f"λ1 = {λ1}")
        print(f"λ2 = {λ2}")
        print(f"g  = {g}")
        print(f"hx = ({hx**(1/3)} MeV)^3")
        print(f"hy = ({hy**(1/3)} MeV)^3")
        print(f"Λ  = {Λ} MeV")

        Δx, Δy, μu, μd, μs, μe = sp.symbols("Δ_x Δ_y μ_u μ_d μ_s μ_e", complex=True)
        σx = 2*Δx/g
        σy = np.sqrt(2)*Δy/g
        Ωb = m2/2*(σx**2+σy**2) + λ1/4*(σx**2+σy**2)**2 + λ2/8*(σx**4+2*σy**4) - hx*σx - hy*σy 
        Ωr = Nc/(16*π**2)*(Δx**4*(3/2+sp.log(Λ**2/Δx**2))+Δx**4*(3/2+sp.log(Λ**2/Δx**2))+Δy**4*(3/2+sp.log(Λ**2/Δy**2)))
        Ωu = -Nc/(24*π**2)*((2*μu**2-5*Δx**2)*μu*sp.sqrt(μu**2-Δx**2)+3*Δx**4*sp.asinh(sp.sqrt(μu**2/Δx**2-1)))
        Ωd = -Nc/(24*π**2)*((2*μd**2-5*Δx**2)*μd*sp.sqrt(μd**2-Δx**2)+3*Δx**4*sp.asinh(sp.sqrt(μd**2/Δx**2-1)))
        Ωs = -Nc/(24*π**2)*((2*μs**2-5*Δy**2)*μs*sp.sqrt(μs**2-Δy**2)+3*Δy**4*sp.asinh(sp.sqrt(μs**2/Δy**2-1)))
        Ωe =  -1/(24*π**2)*((2*μe**2-5*me**2)*μe*sp.sqrt(μe**2-me**2)+3*me**4*sp.asinh(sp.sqrt(μe**2/me**2-1)))

        Ω  = Ωb + Ωr + Ωu + Ωd + Ωs + Ωe
        dΩx = sp.diff(Ω, Δx)
        dΩy = sp.diff(Ω, Δy)

        Ω   = sp.lambdify((Δx, Δy, μu, μd, μs, μe), Ω,   "numpy")
        dΩx = sp.lambdify((Δx, Δy, μu, μd, μs, μe), dΩx, "numpy")
        dΩy = sp.lambdify((Δx, Δy, μu, μd, μs, μe), dΩy, "numpy")
        self.Ω   = lambda Δx, Δy, μu, μd, μs, μe: np.real(  Ω(Δx+0j, Δy+0j, μu+0j, μd+0j, μs+0j, μe+0j))
        self.dΩx = lambda Δx, Δy, μu, μd, μs, μe: np.real(dΩx(Δx+0j, Δy+0j, μu+0j, μd+0j, μs+0j, μe+0j))
        self.dΩy = lambda Δx, Δy, μu, μd, μs, μe: np.real(dΩy(Δx+0j, Δy+0j, μu+0j, μd+0j, μs+0j, μe+0j))

    def solve(self, Δx, guess):
        def system(μQ_Δy_μe):
            μQ, Δy, μe = μQ_Δy_μe # unpack variables
            μu, μd, μs = μelim(μQ, μe)
            return (self.dΩx(Δx, Δy, μu, μd, μs, μe),
                    self.dΩy(Δx, Δy, μu, μd, μs, μe),
                    charge(Δx, Δy, μu, μd, μs, μe))
        sol = scipy.optimize.root(system, guess, method="lm") # lm works, hybr works but unstable for small μ
        assert sol.success, f"{sol.message} (Δx = {Δx})"
        μQ, Δy, μe = sol.x
        μu, μd, μs = μelim(μQ, μe)
        return μQ, Δy, μu, μd, μs, μe

if __name__ == "__main__":
    # plot massive, interacting and massless, free equation of state

    # TODO: remove old data files
    # TODO: make report use new data files

    # plot 3D potential for 2-flavor model with μu=μd
    """
    model = LSM2Flavor()
    Δ = np.linspace(-1000, +1000, 100)
    μQ = np.linspace(0, 500, 50)
    Ω = np.array([model.Ω(Δ, 0, μQ, μQ, 0, 0) for μQ in μQ])
    Δ0 = np.empty_like(μQ)
    Ω0 = np.empty_like(μQ)
    for i in range(0, len(μQ)):
        μQ0 = μQ[i]
        def Ω2(Δ): return model.Ω(Δ, 0, μQ0, μQ0, 0, 0)
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
    utils.writecols(cols, heads, f"data/{model.name}/potential.dat", skipevery=len(μQ))
    """

    # TEST GROUND TODO: remove
    """
    Δ = np.linspace(-600, +600, 300)
    for mσ in [500, 550, 600, 650, 700, 750, 800, 850]:
        LSM2Flavor(mσ=mσ).vacuum_potential(Δ, np.array([ms]), write=True)
    Δ = np.linspace(-1000, +1000, 50)
    for mσ in [500, 550, 600, 650, 700, 750, 800, 850]:
        LSM3Flavor(mσ=mσ).vacuum_potential(Δ, Δ, write=True)
    """

    """
    model = LSM2Flavor(mσ=500) # TODO: need mσ > 600 to avoid starting backwards?
    Δ = np.linspace(-500, +500, 50)
    model.eos(np.linspace(300, 0, 500)[:-1], B=0**4, plot=True)
    model.stars([27, 34, 41, 48], (1e-7, 1e-1), plot=True)

    model = LSM2Flavor() # TODO: 3flavor?
    for Pc in [0.0006, 0.0008, 0.001]:
        model.eos(B=40**4, plot=True)
        model.star(B14=38, Pc=Pc)
    """

    """
    for model in (Bag2Flavor(), Bag3Flavor()):
        μQ = np.linspace(0, 1000, 1000)[1:]
        model.eos(μQ, plot=True, write=True)

        # solve TOV equation for different bag pressures
        Bs = [144, 163]
        model.stars(Bs, (1e-7, 1e1), write=True)
    """

    model = LSM2Flavor(mσ=700)
    Δx = np.linspace(300, 0, 200)[:-1]
    model.eos(Δx, B=30**4, plot=True, write=True)

    #for model in (LSM2Flavor(), LSM2FlavorConsistent(), LSM3Flavor()):
    for model in (LSM2Flavor(mσ=800),):
        Δx = np.linspace(300, 0, 200)[:-1]
        model.eos(Δx, plot=True, write=False)

        # solve TOV equation for different bag pressures
        Bs = [6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132]
        model.stars(Bs, (1e-7, 1e1), write=True)
