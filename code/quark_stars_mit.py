#!/usr/bin/python3

from quark_hybrid_stars_common import *

muf = 5 # current/lone/free masses (i.e. without gluons)
mdf = 7
msf = 150

class MITModel(Model):
    pass # will only inherit

class MIT2FlavorModel(MITModel):
    def __init__(self):
        Model.__init__(self, "MIT2F")
        self.Ω = lambda mu, md, ms, μu, μd, μs, μe: np.real(
            -Nc/(24*π**2)*((2*μu**2-5*mu**2)*μu*np.sqrt(μu**2-mu**2+0j)+\
            3*mu**4*np.arcsinh(np.sqrt(μu**2/mu**2-1+0j))) + \
            -Nc/(24*π**2)*((2*μd**2-5*md**2)*μd*np.sqrt(μd**2-md**2+0j)+\
            3*md**4*np.arcsinh(np.sqrt(μd**2/md**2-1+0j))) + \
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
            -Nc/(24*π**2)*((2*μu**2-5*mu**2)*μu*np.sqrt(μu**2-mu**2)+\
            3*mu**4*np.arcsinh(np.sqrt(μu**2/mu**2-1))) + \
            -Nc/(24*π**2)*((2*μd**2-5*md**2)*μd*np.sqrt(μd**2-md**2)+\
            3*md**4*np.arcsinh(np.sqrt(μd**2/md**2-1))) + \
            -Nc/(24*π**2)*((2*μs**2-5*ms**2)*μs*np.sqrt(μs**2-ms**2)+\
            3*ms**4*np.arcsinh(np.sqrt(μs**2/ms**2-1))) + \
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

if __name__ == "__main__": # uncomment lines/blocks to run
    """
    models = [MIT2FlavorModel, MIT3FlavorModel]
    for model in models:
        model = model()
        model.eos(plot=False, write=True)
        for B14 in (145, 150, 155):
            model.stars(B14, (1e-7, 1e-2), write=True)
    """
