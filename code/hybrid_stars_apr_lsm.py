#!/usr/bin/python3

from quark_hybrid_stars_common import *
import quark_stars_lsm

class HybridModel(Model):
    def eos(self, N=1000, B=111**4, hybrid=True, plot=False, write=False):
        arr = np.loadtxt("data/APR/eos.dat")
        mn = 900 # MeV
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
            ax2.plot(μB1, nB1/0.165, "-b.")
            ax2.plot(μB1, nB2/0.165, "-r.")
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
        return ϵint, nuint, ndint, nsint, lambda x: 0*x, μQint # ignore electrons

class Hybrid2FlavorModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM2F_APR"
        self.mσ = mσ
        self.quarkmodel = quark_stars_lsm.LSM2FlavorModel

class Hybrid3FlavorModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM3F_APR"
        self.mσ = mσ
        self.quarkmodel = quark_stars_lsm.LSM3FlavorModel

class Hybrid2FlavorConsistentModel(HybridModel):
    def __init__(self, mσ=600):
        self.name = "LSM2FC_APR"
        self.mσ = mσ
        self.quarkmodel = quark_stars_lsm.LSM2FlavorConsistentModel

if __name__ == "__main__": # uncomment lines/blocks to run
    #Hybrid2FlavorModel(mσ=600).eos(B=111**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=700).eos(B=68**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=800).eos(B=27**4, plot=True, write=True)
    #Hybrid2FlavorModel(mσ=600).stars(111, (1e-5, 1e-2), write=True)
    #Hybrid2FlavorModel(mσ=700).stars(68,  (1e-5, 1e-2), write=True)
    #Hybrid2FlavorModel(mσ=800).stars(27,  (1e-5, 1e-2), write=True)
    #Hybrid2FlavorModel(mσ=600).star(0.001180703125, 111, write=True)

    #Hybrid3FlavorModel(mσ=600).eos(B=111**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=700).eos(B=68**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=800).eos(B=27**4, plot=True, write=True)
    #Hybrid3FlavorModel(mσ=600).stars(111, (1e-5, 1e-2), write=True)
    #Hybrid3FlavorModel(mσ=700).stars(68,  (1e-5, 1e-2), write=True)
    #Hybrid3FlavorModel(mσ=800).stars(27,  (1e-5, 1e-2), write=True)
    #Hybrid3FlavorModel(mσ=600).star(0.0008160778808593749, 111, write=True)

    #Hybrid2FlavorConsistentModel(mσ=400).eos(B=107**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=500).eos(B=84**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=600).eos(B=27**4, plot=True, write=True)
    #Hybrid2FlavorConsistentModel(mσ=400).stars(107, (1e-5, 1e-2), write=True)
    #Hybrid2FlavorConsistentModel(mσ=500).stars(84,  (1e-5, 1e-2), write=True)
    #Hybrid2FlavorConsistentModel(mσ=600).stars(27,  (1e-5, 1e-2), write=True)
    #Hybrid2FlavorConsistentModel(mσ=400).star(0.001180703125, 107, write=True)
