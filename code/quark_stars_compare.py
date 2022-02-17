#!/usr/bin/python3

import constants
π = constants.π
from tov import massradiusplot
import utils

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

Nc = 3
Nf = 2
mσ = 900 # TODO: increase to make nicer EOS
mπ = 138
mq0 = 300 # TODO: decrease to make nicer EOS
fπ = 93
me = 0.5

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)
Λ = g*fπ / np.sqrt(np.e)

def qf(σ, μu, μd, μe):
    mq = g*σ
    return 2*np.real(np.power(μu**2-mq**2+0j, 3/2)) - np.real(np.power(μd**2-mq**2+0j, 3/2)) - np.real(np.power(μe**2-me**2+0j, 3/2))

def ωf(σ, μu, μd, μe):
    mq = g*σ
    ω0 = -1/2*m**2*σ**2 + λ/24*σ**4 - h*σ + Nc*Nf*mq**4/(16*π**2)*(3/2+np.log(Λ**2/mq**2))
    ωu = -Nc/(24*π**2) * ((2*μu**2-5*mq**2)*μu*np.real(np.sqrt(μu**2-mq**2+0j)) + 3*mq**4*np.arcsinh(np.real(np.sqrt(μu**2/mq**2-1+0j))))
    ωd = -Nc/(24*π**2) * ((2*μd**2-5*mq**2)*μd*np.real(np.sqrt(μd**2-mq**2+0j)) + 3*mq**4*np.arcsinh(np.real(np.sqrt(μd**2/mq**2-1+0j))))
    ωe =  -1/(24*π**2) * ((2*μe**2-5*me**2)*μe*np.real(np.sqrt(μe**2-me**2+0j)) + 3*me**4*np.arcsinh(np.real(np.sqrt(μe**2/me**2-1+0j))))
    return ω0 + ωu + ωd + ωe

def dωf(σ, μu, μd, μe): # partial derivative with respect to σ
    dω0 = -m**2*σ + λ/6*σ**3 - h + Nc*Nf/(16*π**2) * 4*g**4*σ**3*(1+np.log(Λ**2/(g**2*σ**2)))
    dωu = -Nc/24*(12*g**4*σ**3*np.arcsinh(np.sqrt(μu**2/(g**2*σ**2) - 1)) + (5*g**2*σ**2 - 2*μu**2)*g**2*μu*σ/np.sqrt(-g**2*σ**2 + μu**2) - 10*np.sqrt(-g**2*σ**2 + μu**2)*g**2*μu*σ - 3*g**2*μu**2*σ/(np.sqrt(μu**2/(g**2*σ**2) - 1)*np.sqrt(μu**2/(g**2*σ**2))))/π**2 if μu > g*σ else 0
    dωd = -Nc/24*(12*g**4*σ**3*np.arcsinh(np.sqrt(μd**2/(g**2*σ**2) - 1)) + (5*g**2*σ**2 - 2*μd**2)*g**2*μd*σ/np.sqrt(-g**2*σ**2 + μd**2) - 10*np.sqrt(-g**2*σ**2 + μd**2)*g**2*μd*σ - 3*g**2*μd**2*σ/(np.sqrt(μd**2/(g**2*σ**2) - 1)*np.sqrt(μd**2/(g**2*σ**2))))/π**2 if μd > g*σ else 0
    dωe = 0 # independent of σ
    return dω0 + dωu + dωd + dωe

def eos(μu, method=""):
    σ = np.empty(len(μu))
    μd = np.empty(len(μu))
    μe = np.empty(len(μu))
    ω = np.empty(len(μu))
    for i in range(0, len(μu)):
        μu0 = μu[i]

        if method == "partial":
            def system(σμe): σ, μe = σμe; μd = μu0 + μe; return (dωf(σ, μu0, μd, μe), qf(σ, μu0, μd, μe))
            guess = (σ[i-1], μe[i-1]) if i > 0 else (fπ, 0) # guess root with previous solution
            sol = scipy.optimize.root(system, guess, method="hybr")
            assert sol.success, f"{sol.message} (μu = {μu0})"
            σ0, μe0 = sol.x
            μd0 = μu0 + μe0
        elif method == "total":
            def μelim(σ0):
                if g*σ0 > μu0: # trivial solution: zero density and zero charge
                    μe0 = me
                    μd0 = μe0 + μu0
                else: # non-trivial solution
                    def qf2(μe0): μd0 = μu0 + μe0; return qf(σ0, μu0, μd0, μe0)
                    sol = scipy.optimize.root_scalar(qf2, bracket=(0, 1000), method="brentq")
                    assert sol.converged, f"{sol.message} (σ = {σ0}, μu = {μu0})"
                    μe0 = sol.root
                    μd0 = μu0 + μe0
                return μu0, μd0, μe0

            def ωf2(σ0):
                μu0, μd0, μe0 = μelim(σ0)
                return ωf(σ0, μu0, μd0, μe0)

            sol = scipy.optimize.minimize_scalar(ωf2, bounds=(1, 150), method="bounded")
            assert sol.success, f"{sol.message} (μu = {μu0})"
            σ0 = sol.x
            μu0, μd0, μe0 = μelim(σ0)
        else:
            assert False, f"unknown method: \"{method}\""

        σ[i], μd[i], μe[i], ω[i] = σ0, μd0, μe0, ωf(σ0, μu0, μd0, μe0)
    P = -ω
    P -= P[0]
    mq = g*σ
    nu = np.real(np.power(μu**2-mq**2+0j, 3/2)) / (3*π**2)
    nd = np.real(np.power(μd**2-mq**2+0j, 3/2)) / (3*π**2)
    ne = np.real(np.power(μe**2-me**2+0j, 3/2)) / (3*π**2)
    ϵ = -P + μu*nu + μd*nd + μe*ne
    
    return σ, μu, μd, μe, nu, nd, ne, P, ϵ

μu = np.linspace(250, 400, 500)
σ1, μu1, μd1, μe1, nu1, nd1, ne1, P1, ϵ1 = eos(μu, method="partial")
σ2, μu2, μd2, μe2, nu2, nd2, ne2, P2, ϵ2 = eos(μu, method="total")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
ax1.set_xlabel(r"$\mu_u$")
ax1.set_ylabel(r"$\sigma$")
ax1.plot(μu1, σ1, "-k")
ax1.plot(μu2, σ2, "--k")
ax2.set_xlabel(r"$\mu_u$")
ax2.set_ylabel(r"$n$")
ax2.plot(μu1, nu1, "-r")
ax2.plot(μu1, nd1, "-g")
ax2.plot(μu1, ne1, "-b")
ax2.plot(μu2, nu2, "--r")
ax2.plot(μu2, nd2, "--g")
ax2.plot(μu2, ne2, "--b")
ax3.set_xlabel(r"$P$")
ax3.set_ylabel(r"$\epsilon$")
ax3.plot(P1, ϵ1, "-k")
ax3.plot(P2, ϵ2, "--k")
plt.show()

q1 = qf(σ1, μu1, μd1, μe1)
q2 = qf(σ2, μu2, μd2, μe2)
plt.plot(μu1, np.sign(q1)*np.log(1+np.abs(q1)), "-k")
plt.plot(μu2, np.sign(q2)*np.log(2+np.abs(q2)), "--k")
plt.show()
