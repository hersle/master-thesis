#!/usr/bin/python3

# TODO: use splrep (splines) to interpolate equation of state instead of linear interpolation?

from constants import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize

Nc = 3
Nf = 2
mσ = 550
mπ = 138
mq0 = 300
fπ = 93
me = 0

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

def mq(σ): return g*(σ) # TODO: abs or not?
def x(σ,μ): return np.real(np.sqrt(μ**2 - mq(σ)**2) / mq(σ))

def μelim(σ, μu, μdmax=1e3, verbose=False):
    def f(μd):
        μe = μd - μu # chemical equilibrium
        return 2/3 * (μu**2-mq(σ)**2)**(3/2) - 1/3 * (μd**2-mq(σ)**2)**(3/2) - 1 * (μe**2-me**2)**(3/2) # charge neutrality
    def df(μd): # derivative of f
        μe = μd - μu
        return -μd*(μd**2-mq(σ)**2)**(1/2) - 3*μe*(μe**2-me**2)**(1/2)

    # first find extremum (practical to use as one endpoint in bisection method later)
    #μd = scipy.optimize.root_scalar(df, bracket=(mq(σ), μdmax), method="bisect").root
    #if verbose:
        #print(f"df(μd = {μd}) = {df(μd)}")

    # first find extremum (practical to use as one endpoint in bisection method later)
    μd = scipy.optimize.minimize_scalar(lambda μd: -f(μd), bounds=(mq(σ), μdmax), method="bounded").x
    if verbose:
        print(f"df(μd = {μd}) = {df(μd)}")

    # find leftmost root (not always present!)
    #μD = np.linspace(mq(σ), μd, 500)
    #μd = scipy.optimize.root_scalar(f, bracket=(mq(σ), μd), method="bisect").root
    #print(f"f(μd = {μd}) = {f(μd)}")

    # find rightmost root (always present?)
    μD = np.linspace(mq(σ), μdmax, 500) # for possible plotting
    # TODO: find high enough right bracket?
    μd = scipy.optimize.root_scalar(f, bracket=(μd, μdmax), method="bisect").root
    if verbose:
        print(f"f(μd = {μd}) = {f(μd)}")

    μe = μd - μu

    return μu, μd, μe

def ω(σ, μu, verbose=True):
    if type(σ) == np.ndarray:
        return np.array([ω(σ, μu, verbose=verbose) for σ in σ])
    if verbose:
        print(f"ω(σ={σ}, μu={μu}); mq = {mq(σ)}")

    μu, μd, μe = μelim(σ, μu)

    ω0 = -1/2*m**2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq(σ)**4/(16*π**2)*(3/2+np.log(Λ**2/mq(σ)**2))
    ωe = -μe**4 / (12*π**2)
    ωu = -Nc/(24*π**2) * ((2*μu**2-5*mq(σ)**2)*μu*np.sqrt(μu**2-mq(σ)**2) + 3 * mq(σ)**4 * np.arcsinh(np.sqrt(μu**2/mq(σ)**2-1)))
    #xu = x(σ,μu)
    #ωu = -Nc*mq(σ)**4/(24*π**2) * ((2*x(σ,μu)**3-3*x(σ,μu))*np.sqrt(x(σ,μu)**2+1) + 3*np.arcsinh(x(σ,μu)))
    return (ω0 + ωe + ωu) / fπ**4

def minσ(μu):
    return np.array([scipy.optimize.minimize_scalar(ω, bounds=(0, μu[0]/g), method="bounded", args=(μ)).x for μ in μu])

μu = np.linspace(280, 400, 500)
σ0 = minσ(μu)
μ = np.array([μelim(σ0, μu) for σ0, μu in zip(σ0, μu)])
μu0, μd0, μe0 = μ[:,0], μ[:,1], μ[:,2]

plt.plot(μu, σ0)
plt.plot(μu, μu/g)
plt.show()

print(len(σ0), len(μu0), len(μd0), len(μe0))

"""
σ = np.linspace(0, μ[0]/g, 100)[1:-1]
#ωs = np.array([[ω(σ,μ) for μ in μ] for σ in σ])
fig, ax = plt.subplots()
for j in range(0, len(μu)):
    ax.plot(σ, ω(σ, μu[j]))
    ax.scatter(σ0[j], ω(σ0[j], μu[j]))
plt.show()
"""

ω0 = np.array([ω(σ0[i], μu0[i]) for i in range(0, len(μu0))])
#ω0 -= fπ / fπ**4 # TODO: subtract from known vacuum value?
print(ω0.shape)
P0 = -ω0
ne = (μe0**2-me**2)**(3/2) / (3*π**2)
nu = (μu0**2-mq(σ0)**2)**(3/2) / (3*π**2)
nd = (μd0**2-mq(σ0)**2)**(3/2) / (3*π**2)
ϵ0 = (μe0*ne + μu0*nu + μd0*nd) / fπ**4 - P0

plt.plot(nu, P0)
plt.plot(nu, ϵ0)
plt.show()

plt.plot(P0, ϵ0, "-k")
plt.show()

P0 -= np.interp(0, ϵ0, P0) # subtract (interpolated) value P(ϵ=0) so that the new pressure has P(ϵ=0)=0

# TODO: remove negative values of P (and corresponding ones of ϵ), then add (0,0)
P0 = np.insert(P0[ϵ0 >= 0], 0, 0)
ϵ0 = np.insert(ϵ0[ϵ0 >= 0], 0, 0)
print(P0[0], ϵ0[0])

plt.plot(P0, ϵ0, "-k")
plt.xlabel(r"$P$")
plt.ylabel(r"$\epsilon$")
plt.show()
