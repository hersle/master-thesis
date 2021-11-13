from constants import *
import matplotlib.pyplot as plt
import numpy as np

# TODO: np.gradient needs r parameter?

def shoot(r, Π, Q, W, ω2, j=50):
    dΠdr = np.gradient(Π, r)
    F = Q + ω2*W

    """
    plt.plot(r[j:], 1/Π[j:])
    plt.ylim(-100, +100)
    plt.show()
    """

    def Πf(n):
        n1 = int(n)
        n2 = n1 + 1
        R = r[n1] + (r[n2]-r[n1])*(n-n1)
        return Π[n1] + (Π[n2]-Π[n1]) * (R - r[n1]) / (r[n2]-r[n1])
        #return np.interp(r[n1] + (n-n1)*(r[n2]-r[n1]), r, Π)

    u = np.empty(np.shape(r))
    u[0:j] = 100000 * r[0:j]**3
    for i in range(j-1, len(r)-1):
        hp = r[i+1] - r[i]
        hm = r[i] - r[i-1]
        H = hp + hm
        """
        if np.abs(u[i]) > 1e4:
            u[i+1] = u[i] + np.sign(u[i])
        """
        denom = 2*Π[i]/hp+dΠdr[i]
        factor = 1/denom
        #print(factor)
        if np.abs(factor) > 1e2:
            # "handle" blowup without oscillating back
            u[i+1] = u[i] + (u[i]-u[i-1]) * (r[i+1]-r[i]) / (r[i]-r[i-1]) # linterp last few points
        else:
            u[i+1] = factor * ( u[i] * (2*Π[i]*(1/hp+1/hm) - H*(Q[i]+ω2*W[i])) + u[i-1] * (dΠdr[i] - 2*Π[i]/hm) )
            #u[i+1] = 1 / (2*Πf(i+1/2)*hm) * ( u[i] * (2*Πf(i+1/2)*hm + 2*Πf(i-1/2)*hp - F[i]*H*hp*hm) + u[i-1] * (2*Πf(i-1/2)*hp) )
        #print(u[i+1]-u[i])
    #print(u[-10:])
    return u, count_nodes(u)

def count_nodes(u):
    nodes = 0
    for i in range(1, len(u)):
        u1 = u[i-1]
        u2 = u[i]
        if u1 * u2 < 0:
            nodes += 1
    return nodes

def search(r, Π, Q, W, N=0):
    def increaseuntil(ω20, cond, sign=+1):
        ω2 = ω20
        u, n = shoot(r, Π, Q, W, ω2)
        while not cond(u, n):
            ω2 += sign*1 if ω2 == ω20 else (ω2 - ω20) # exponential increase
            u, n = shoot(r, Π, Q, W, ω2)
            #plt.plot(r, u)
            #plt.show()
        return ω2

    def decreaseuntil(ω20, cond):
        return increaseuntil(ω20, cond, sign=-1)

    def bisectuntil(ω21, ω22, tol=1e-5, plot=False):
        u1, n1 = shoot(r, Π, Q, W, ω21)
        u2, n2 = shoot(r, Π, Q, W, ω22)
        while ω22 - ω21 > tol:
            ω23 = (ω21 + ω22) / 2
            u3, n3 = shoot(r, Π, Q, W, ω23)
            print(f"ω2 = {ω23} -> {n3} nodes")
            if plot:
                plt.plot(r, u3, label=f"n = {n3}, ω2 = {ω23}")
            if n3 > N:
                ω22, u2, n2 = ω23, u3, n3
            else:
                ω21, u1, n1 = ω23, u3, n3
        ω2 = (ω21 + ω22) / 2
        u, n = shoot(r, Π, Q, W, ω23)
        if plot:
            plt.ylim(-1e3, +1e3)
            plt.legend()
            plt.show()
        return ω2, u, n

    ω20 = 0
    u, n = shoot(r, Π, Q, W, ω20)
    if n > N:
        # ω20 is an upper bound
        ω22 = ω20
        ω21 = decreaseuntil(ω20, lambda u, n: n <= N)
    else:
        # ω20 is a lower bound
        ω21 = ω20
        ω22 = increaseuntil(ω20, lambda u, n: n > N)

    """ 
    #DEBUG
    print(ω21, ω22)
    u1, n1 = shoot(r, Π, Q, W, ω21)
    u2, n2 = shoot(r, Π, Q, W, ω22)
    plt.plot(r, u1, label=f"{n1}, {ω21}")
    plt.plot(r, u2, label=f"{n2}, {ω22}")
    plt.legend()
    plt.ylim(-1e2, +1e2)
    plt.show()
    """

    ω2, u, n = bisectuntil(ω21, ω22, plot=True)
    return ω2, u, n


def fundamental_mode(r, m, P, α, ϵ):
    dPdr = np.gradient(P, r)
    dPdϵ = np.gradient(P, ϵ)

    β = -1/2*np.log(1-2*G*m/r) # β(0) = 0, avoid division by 0, already dimensionless
    Γ = (P + ϵ) / P * dPdϵ # already dimensionless
    Π = np.exp(β+3*α)/r**2 * Γ * P
    Q = -4*np.exp(β+3*α)/r**3*dPdr - 8*π*(G/4*π)*np.exp(3*β+3*α)/r**2*P*(ϵ+P) + np.exp(β+3*α)*dPdr**2 / (r**2*(ϵ+P))
    #Q[-10:] = 0 # TODO: reliably set last Qs to 0 ?
    W = np.exp(3*β+α)*(ϵ+P)/r**2

    # TODO: can i divide Π, Q, W by their maximums to make numbers more handleable?

    """
    # TRY OUT
    ω2 = np.linspace(-5, +5, 10)

    for ω2 in ω2:
        u, n = shoot(r, Π, Q, W, ω2)
        plt.plot(r, u, marker="o", label=f"{n}, {ω2:.2f}")
    plt.ylim(-1e3, +1e3)
    plt.legend(loc="upper left")
    plt.show()
    """

    #plt.plot(r[50:], Q[50:])
    #plt.show()

    ω2, u, n = search(r, Π, Q, W)
    return ω2, u
