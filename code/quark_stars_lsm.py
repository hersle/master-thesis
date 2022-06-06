#!/usr/bin/python3

from quark_hybrid_stars_common import *
import os; os.environ["XDG_SESSION_TYPE"] = "x11" # for mayavi to work
import mayavi.mlab as mlab

mu0 = 300 # constituent masses (i.e. with gluons)
md0 = mu0
ms0 = 429 # only used for root equation guess

fπ = 93
fK = 113
σx0 = fπ
σy0 = np.sqrt(2)*fK-fπ/np.sqrt(2)

mσ = 800
mπ = 138
mK = 496

class LSMModel(Model):
    def eossolve(self, N):
        Δx = np.linspace(self.mu, 0, N)[:-1] # shave off erronous 0
        Δy = np.empty_like(Δx)
        μQ, μu, μd, μs, μe = Δy, Δy, Δy, Δy, Δy # copy

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
            return (self.dΩ(Δx, Δx, 0, μu, μd, 0, μe), Δy, # hack to give Δy = 0
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
        def mesonmasses(m2, λ1, λ2):
            rt2 = np.sqrt(2)

            m2σσ00 = m2+λ1/3*(4*rt2*σx0*σy0+7*σx0**2+5*σy0**2)+λ2*(σx0**2+σy0**2)
            m2σσ11 = m2+λ1*(σx0**2+σy0**2)+3/2*λ2*σx0**2
            m2σσ44 = m2+λ1*(σx0**2+σy0**2)+λ2/2*(rt2*σx0*σy0+σx0**2+2*σy0**2)
            m2σσ88 = m2-λ1/3*(4*rt2*σx0*σy0-5*σx0**2-7*σy0**2)+λ2/2*(σx0**2+4*σy0**2)
            m2σσ08 = 2/3*λ1*(rt2*σx0**2-rt2*σy0**2-σx0*σy0)+λ2/rt2*(σx0**2-2*σy0**2)

            m2ππ00 = m2+λ1*(σx0**2+σy0**2)+λ2/3*(σx0**2+σy0**2)
            m2ππ11 = m2+λ1*(σx0**2+σy0**2)+λ2/2*σx0**2
            m2ππ44 = m2+λ1*(σx0**2+σy0**2)-λ2/2*(rt2*σx0*σy0-σx0**2-2*σy0**2)
            m2ππ88 = m2+λ1*(σx0**2+σy0**2)+λ2/6*(σx0**2+4*σy0**2)
            m2ππ08 = λ2/6*(rt2*σx0**2-2*rt2*σy0**2)

            θσ = np.arctan(2*m2σσ08 / (m2σσ88-m2σσ00)) / 2
            θπ = np.arctan(2*m2ππ08 / (m2ππ88-m2ππ00)) / 2

            m2f0 = m2σσ00*np.sin(θσ)**2 + m2σσ88*np.cos(θσ)**2 + m2σσ08*np.sin(2*θσ)
            m2σ  = m2σσ00*np.cos(θσ)**2 + m2σσ88*np.sin(θσ)**2 - m2σσ08*np.sin(2*θσ)
            m2a0 = m2σσ11
            m2κ  = m2σσ44
            m2η  = m2ππ00*np.sin(θπ)**2 + m2ππ88*np.cos(θπ)**2 + m2ππ08*np.sin(2*θπ)
            m2ηp = m2ππ00*np.cos(θπ)**2 + m2ππ88*np.sin(θπ)**2 - m2ππ08*np.sin(2*θπ)
            m2π  = m2ππ11
            m2K  = m2ππ44
            return m2f0, m2σ, m2a0, m2κ, m2η, m2ηp, m2π, m2K

        def system(m2_λ1_λ2):
            m2, λ1, λ2 = m2_λ1_λ2
            _, m2σ, _, _, _, _, m2π, m2K = mesonmasses(m2, λ1, λ2)
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
            Λx, Λy = Λ, Λ # set common, averaged renormalization scale
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

        # predict (remaining) meson masses
        m2f0, m2σ, m2a0, m2κ, m2η, m2ηp, m2π, m2K = mesonmasses(m2, λ1, λ2)
        print(f"mf0 = {np.sqrt(m2f0):.0f} MeV")
        print(f"mσ  = {np.sqrt(m2σ):.0f} MeV")
        print(f"ma0 = {np.sqrt(m2a0):.0f} MeV")
        print(f"mκ  = {np.sqrt(m2κ):.0f} MeV")
        print(f"mη  = {np.sqrt(m2η):.0f} MeV")
        print(f"mηp = {np.sqrt(m2ηp):.0f} MeV")
        print(f"mπ  = {np.sqrt(m2π):.0f} MeV")
        print(f"mK  = {np.sqrt(m2K):.0f} MeV")

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

if __name__ == "__main__": # uncomment lines/blocks to run
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
    """

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
