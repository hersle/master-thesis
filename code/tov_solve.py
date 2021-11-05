#!/usr/bin/python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import utils

from constants import c
m0 = 1.98847e30 # kg
r0 = 10e3 # m
G = 6.67430e-11 / (r0*c**2/m0) # dimensionless

def soltov(ϵ, P0, maxdr=1e-3):
    print(f"Solving TOV with ϵ = {ϵ.__name__}, P0 = {P0}, maxdr = {maxdr} ...")

    def rhs(r, y):
        m, P = y[0], y[1]
        dmdr = 3*r**2*ϵ(P)
        if r == 0:
            dPdr = 0 # avoid division by r = 0 (m = 0 implies dPdr = 0)
        else:
            dPdr = -G*m*ϵ(P)/r**2 * (1 + P/ϵ(P)) * (1 + 3*r**2*P/ϵ(P)) / (1 - 2*G*m/r)
        return [dmdr, dPdr]

    def terminator(r, y):
        m, P = y[0], y[1]
        return P - 0
    terminator.terminal = True # stop integration when P == 0, use as last point

    res = scipy.integrate.solve_ivp(
        rhs, (0, np.inf), (0, P0), events=terminator, max_step=maxdr
    )
    assert res.success, "ERROR: " + res.message

    rs, ms, Ps = res.t, res.y[0,:], res.y[1,:]
    return rs, ms, Ps
    
def massradius(ϵ, P0, maxdr=1e-3):    
    rs, ms, Ps = soltov(ϵ, P0, maxdr=maxdr)
    R, M = rs[-1], ms[-1]
    return R, M

# Bisect [P1, P2] to make points evenly
def massradiusplot(ϵ, P1P2, tolD=1e-5, tolP=1e-6, maxdr=1e-3, outfile="", visual=False):
    P1, P2 = P1P2[0], P1P2[1]
    R1, M1 = massradius(ϵ, P1, maxdr=maxdr)
    R2, M2 = massradius(ϵ, P2, maxdr=maxdr)
    Ps, Ms, Rs = [P1, P2], [M1, M2], [R1, R2]

    if visual:
        plt.ion() # automatically update open figure
        graph, = plt.plot([], [], "k-o") # modify graph data later

    i = 0
    while i < len(Ps) - 1:
        P1, M1, R1 = Ps[i],   Ms[i],   Rs[i]
        P2, M2, R2 = Ps[i+1], Ms[i+1], Rs[i+1]

        # Split intervals based on Euclidean distance between (R, M)-points in plot
        # But make sure P1, P2 do not get too close, otherwise algorithm gets stuck
        D = np.sqrt((R1 - R2)**2 + (M1 - M2)**2)
        if D > tolD and P2 - P1 > tolP:
            # split [P1, P2] into [P1, (P1+P2)/2] and [(P1+P2)/2, P2]
            P3 = (P1 + P2) / 2
            R3, M3 = massradius(ϵ, P3, maxdr=maxdr)
            Ps.insert(i+1, P3)
            Ms.insert(i+1, M3)
            Rs.insert(i+1, R3)

            if visual:
                # Animate plot in real-time for immediate feedback
                # inspired by https://stackoverflow.com/a/10944967
                graph.set_data(Rs, Ms)
                plt.gca().relim() # autoscale only works in animation after this
                plt.autoscale()
                plt.draw()
                plt.pause(0.001)
        else:
            i += 1

    if visual:
        plt.ioff() # leave original state
        plt.show() # leave final plot open

    if outfile != "":
        utils.writecols([Ps, Ms, Rs], ["P", "M", "R"], outfile)
        print(f"Wrote (P, M, R) to {outfile}")
    
    return Ps, Ms, Rs
