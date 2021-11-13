from constants import *
import matplotlib.pyplot as plt
import numpy as np

def shoot(r, Π, Q, W, ω2, p1, p2):
    dΠdr = np.gradient(Π, r)
    R = r[-1]

    u = np.empty(np.shape(r))

    # Start: set boundary condition u ∝ r^3
    i = 0
    while r[i] / R < p1:
        u[i] = 1 * r[i]**3
        i += 1

    # Middle: integrate Sturm-Liouville equation
    i -= 1 # i in next loop is shifted by -1
    while r[i] / R < p2:
        hp = r[i+1] - r[i]
        hm = r[i] - r[i-1]
        H = hp + hm
        factorp = 2 * Π[i] / hp + dΠdr[i]
        factor0 = 2 * Π[i] * (1/hp + 1/hm) - H * (Q[i] + ω2 * W[i])
        factorm = dΠdr[i] - 2*Π[i]/hm
        u[i+1] = 1/factorp * (u[i] * factor0 + u[i-1] * factorm)
        i += 1

    # End: linearly interpolate to capture divergence without numerical error
    while i < len(r) - 1:
        u[i+1] = u[i] + (u[i]-u[i-1]) * (r[i+1]-r[i]) / (r[i]-r[i-1])
        i += 1

    nodes = sum(u[1:] * u[:-1] < 0) # nodes where u[i] * u[i-1] < 0
    return u, nodes

def search(r, Π, Q, W, N, p1, p2, plot=False, progress=False):
    def increaseuntil(ω20, cond, sign=+1):
        ω2 = ω20
        u, n = shoot(r, Π, Q, W, ω2, p1, p2)
        while not cond(u, n):
            ω2 += sign*1 if ω2 == ω20 else (ω2 - ω20) # exponential increase
            u, n = shoot(r, Π, Q, W, ω2, p1, p2)
        return ω2

    def decreaseuntil(ω20, cond):
        return increaseuntil(ω20, cond, sign=-1)

    def bisectuntil(ω21, ω22, tol=1e-8, plot=False, progress=False):
        u1, n1 = shoot(r, Π, Q, W, ω21, p1, p2)
        u2, n2 = shoot(r, Π, Q, W, ω22, p1, p2)
        i = 0
        us = []
        while ω22 - ω21 > tol:
            ω23 = (ω21 + ω22) / 2
            u3, n3 = shoot(r, Π, Q, W, ω23, p1, p2)
            if progress:
                print(f"\rShooting with ω2 = {ω23:.15f} -> {n3:3d} nodes", end="")
            us.append(u3)
            if n3 > N:
                ω22, u2, n2 = ω23, u3, n3
            else:
                ω21, u1, n1 = ω23, u3, n3
            i += 1
        ω2 = (ω21 + ω22) / 2
        u, n = shoot(r, Π, Q, W, ω23, p1, p2)

        if progress:
            print() # newline
        if plot:
            for i in range(0, len(us)):
                plt.plot(r, us[i], color=(i/len(us), 0, 0))

            # find first index (from back) where derivative is zero
            i = len(u) - 2
            du = np.gradient(u)
            while i >= 0 and du[i] * du[i+1] > 0:
                i -= 1
            ymax = u[i]
            plt.ylim(-2*ymax, +2*ymax)
            plt.show()

        return ω2, u, n

    ω20 = 0
    u, n = shoot(r, Π, Q, W, ω20, p1, p2)
    if n > N:
        # ω20 is an upper bound, search for a lower bound
        ω22 = ω20
        ω21 = decreaseuntil(ω20, lambda u, n: n <= N)
    else:
        # ω20 is a lower bound, search for an upper bound
        ω21 = ω20
        ω22 = increaseuntil(ω20, lambda u, n: n > N)

    ω2, u, n = bisectuntil(ω21, ω22, plot=plot, progress=progress)
    return ω2, u, n


def eigenmode(r, m, P, α, ϵ, N, p1=0.01, p2=0.99, plot=False, progress=False):
    dPdr = np.gradient(P, r)
    dPdϵ = np.gradient(P, ϵ)

    β = -1/2*np.log(1-2*G*m/r) # β(0) = 0, avoid division by 0, already dimensionless
    Γ = (P + ϵ) / P * dPdϵ # already dimensionless
    Π = np.exp(β+3*α)/r**2 * Γ * P
    Q = -4*np.exp(β+3*α)/r**3*dPdr - 8*π*(G/4*π)*np.exp(3*β+3*α)/r**2*P*(ϵ+P) + np.exp(β+3*α)*dPdr**2 / (r**2*(ϵ+P))
    W = np.exp(3*β+α)*(ϵ+P)/r**2

    # TODO: can i divide Π, Q, W by their maximums to make numbers more handleable?

    ω2, u, n = search(r, Π, Q, W, N, p1, p2, plot=plot, progress=progress)
    return ω2, u
