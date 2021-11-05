#!/usr/bin/python3

import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

# constants
π = np.pi
m0 = 1.98847e30 # kg
c = 299792458 # m/s
r0 = 10e3 # m
m = 1.67492749804e-27 # kg
ħ = 1.054571817e-34 # Js 
ratio = m**8*c**6*r0**6 / (m0**2*ħ**6)

ϵ0 = m0*c**2 / (4/3*π*r0**3)
print(f"ϵ0 = {ϵ0}")

kB = 1.380649e-23
T = 1e6 # https://en.wikipedia.org/wiki/Neutron_star#Mass_and_temperature
print(f"m c^2 / k_B T = {m * c**2 / (kB * T)} >> 1")


G = 6.67430e-11 # dimensionful
G = G / (r0*c**2/m0) # dimensionless

def writecols(cols, headers, filename):
    maxlen = max(len(col) for col in cols)
    for col in cols:
        while len(col) < maxlen:
            col.append(np.nan)
        
    file = open(filename, "w")
    file.write(" ".join(headers) + "\n")
    for r in range(0, maxlen):
        for col in cols:
            file.write(str(col[r]) + " ")
        file.write("\n")
    file.close()

def soltov(ϵ, P0, max_step=1e-3):
    def f(r, y):
        m, P = y[0], y[1]
        dmdr = 3*r**2*ϵ(P)
        dPdr = -G*m*ϵ(P)/r**2 * (1 + P/ϵ(P)) * (1 + 3*r**2*P/ϵ(P)) / (1 - 2*G*m/r) if r != 0 else 0 # avoid division by r=0 (correct because m=0 when r=0)
        return [dmdr, dPdr]

    # stop integration when P == 0 and use this as last point
    def e(r, y):
        m, P = y[0], y[1]
        return P - 0
    e.terminal = True
    e.direction = -1

    res = scipy.integrate.solve_ivp(f, (0, np.inf), (0, P0), events=e, max_step=max_step)
    assert res.success, "ERROR: " + res.message
    rs, ms, Ps = res.t, res.y[0,:], res.y[1,:]

    return rs, ms, Ps

def ϵUR(P): return 3*P

def ϵNR(P): return (5**3*4**2 / (3**4*π**2) * ratio * np.abs(P)**3)**(1/5)

# Px(1e7) ≈ 1e28, so need to consider x ∈ [0, 1e7] for P ∈ [0, 1e28]
def Px(x): return m**4*c**3*r0**3 / (18*π*m0*ħ**3) * ((2*x**3 - 3*x) * np.sqrt(x**2 + 1) + 3*np.arcsinh(x))
def ϵGR(P):
    P = np.abs(P) # TODO: implications?
    def f(x): return Px(x) - P
    sol = scipy.optimize.root_scalar(f, method="bisect", bracket=(0, 1e18))
    assert sol.converged, "ERROR: equation of state root finder did not converge"
    x = sol.root
    ϵ = m**4*c**3*r0**3 / (6*π*m0*ħ**3) * ((2*x**3+x) * np.sqrt(x**2 + 1) - np.arcsinh(x))
    return ϵ
    
def massradius(ϵ, P0, max_step=1e-3):    
    rs, ms, Ps = soltov(ϵ, P0, max_step=max_step)
    R, M = rs[-1], ms[-1]
    return R, M

# Bisect [P1, P2] to make points evenly
def massradiusplot(ϵ, P1, P2, tolD=1e-5, tolP=1e-6, max_step=1e-3, visual=False):
    R1, M1 = massradius(ϵ, P1, max_step=max_step)
    R2, M2 = massradius(ϵ, P2, max_step=max_step)
    Ps, Ms, Rs = [P1, P2], [M1, M2], [R1, R2]

    if visual:
        plt.ion() # automatically update open figure
        graph, = plt.plot([], [], "k-o") # modify graph data later

    i = 0
    while i < len(Ps) - 1:
        P1, M1, R1 = Ps[i],   Ms[i],   Rs[i]
        P2, M2, R2 = Ps[i+1], Ms[i+1], Rs[i+1]
        D = np.sqrt((R1 - R2)**2 + (M1 - M2)**2)

        progress = int((P1 - Ps[0]) / (Ps[-1] - Ps[0]) * 100) # fraction of pressures covered
        print(f"\r{progress:0d} %", end="") # print progress

        if D > tolD and P2 - P1 > tolP:
            # split [P1, P2] into [P1, (P1+P2)/2] and [(P1+P2)/2, P2]
            P3 = (P1 + P2) / 2
            R3, M3 = massradius(ϵ, P3, max_step=max_step)
            Ps.insert(i+1, P3)
            Ms.insert(i+1, M3)
            Rs.insert(i+1, R3)
            if visual:
                # animate, inspired by https://stackoverflow.com/a/10944967
                graph.set_data(Rs, Ms)
                plt.gca().relim() # autoscale only works in animation after this
                plt.autoscale()
                plt.draw()
                plt.pause(0.001)
        else:
            i += 1

    print("\r100 %") # end progress line
    if visual:
        plt.ioff() # leave original state
        plt.show() # leave final plot open

    return Ps, Ms, Rs

# rs, ms, Ps = soltov(ϵNR, 2.4)

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-6, 1e10, tolD=0.05, tolP=1e-5, max_step=5e-4, visual=True) # for full curve

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e1, 1e21, tolD=0.04, tolP=1e-3, max_step=2e-4, visual=True) # for spiral only

#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-6, 1e21, tolD=0.05, tolP=1e-5, max_step=2e-4, visual=True) # everything?
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/nr.dat")

# TODO: P never 0, cannot integrate to np.inf
#Ps, Ms, Rs = massradiusplot(ϵUR, 1e-0, 1e4, tolD=0.1, tolP=1e-1, max_step=1e-2, visual=True) # everything?
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/ur.dat")

# Everything for GR
#Ps, Ms, Rs = massradiusplot(ϵGR, 1e-6, 1e17, tolD=0.05, tolP=1e-5, max_step=2e-3, visual=True)
#writecols([Ps, Ms, Rs], ["P", "M", "R"], "data/gr.dat")


#plt.plot(Rs, Ms, "-ko")
#plt.show()
#Ps, Ms, Rs = massradiusplot(ϵNR, 1e-2, 1e2, tolD=0.5, tolP=1e-2, max_step=1e-3, visual=False) # for spiral
