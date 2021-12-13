#!/usr/bin/python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.colors
import utils
from stability import eigenmode

from constants import *

def soltov(ϵ, P0, maxdr=1e-3, progress=True, newtonian=False):
    def printprogress(r, m, P, message="", end=""):
        print(f"\r", end="") # reset line
        print(f"Solving TOV: ", end="")
        print(f"ϵ = {ϵ.__name__}, ", end="")
        print(f"Newtonian={newtonian}, ", end="")
        print(f"P0 = {P0:9.2e}, maxdr = {maxdr:9.2e}, ", end="")
        print(f"r = {r:8.5f}, m = {m:8.5f}, P/P0 = {P/P0:8.5f}", end="")
        if message != "":
            print(f", {message}", end="")
        print("", end=end, flush=True) # flush without newline

    def rhs(r, y):
        m, P, α = y[0], y[1], y[2]
        if progress:
            printprogress(r, m, P)
        E = ϵ(P) # computation can be expensive, only do it once
        dmdr = b*r**2*E
        if r == 0:
            dPdr = 0 # avoid division by r = 0 (m = 0 implies dPdr = 0)
            dαdr = 0
        else:
            if newtonian:
                dPdr = -G*E*m/r**2
            else:
                dPdr = -G/r**2 * (E + P) * (m + b*r**3*P) / (1 - 2*G*m/r)
            #dαdr = (m + 4*π*r**3*P) / (r*(r-2*m))
            if newtonian:
                dαdr = 0
            else:
                dαdr = -dPdr / (E + P)
        return np.array([dmdr, dPdr, dαdr])

    def terminator(r, y):
        m, P, α = y[0], y[1], y[2]
        return P - 0
    terminator.terminal = True # stop integration when P == 0, use as last point

    r1, r2 = 0, np.inf
    res = scipy.integrate.solve_ivp(
        rhs, (0, np.inf), (0, P0, 0), events=terminator, max_step=maxdr
    )
    assert res.success, "ERROR: " + res.message
    rs, ms, Ps, αs = res.t, res.y[0,:], res.y[1,:], res.y[2,:]

    # match α to the Schwarzschild metric at the surface Glendenning (2.226)
    αs = αs - αs[-1] + 1/2 * np.log(1-2*G*ms[-1]/rs[-1]) 

    if progress: # finish progress printer with newline
        printprogress(rs[-1], ms[-1], Ps[-1], res.message, end="\n")

    ϵs = np.array([ϵ(P) for P in Ps]) # (can compute more efficiently than this)

    return rs, ms, Ps, αs, ϵs

# Bisect [P1, P2] to make points evenly
def massradiusplot(
    ϵ, P1P2, tolD=1e-5, tolP=1e-6, maxdr=1e-3, stability=False, newtonian=False,
    outfile="", visual=False
):
    def solvestar(P0):
        rs, ms, Ps, αs, ϵs = soltov(ϵ, P0, maxdr=maxdr, newtonian=newtonian)
        R, M = rs[-1], ms[-1]
        if stability:
            nu = 0
            while eigenmode(rs, ms, Ps, αs, ϵs, nu)[0] < 0:
                nu += 1 # count number of unstable modes with ω2 < 0
        else:
            nu = np.nan
        return R, M, nu

    P1, P2 = P1P2[0], P1P2[1]
    R1, M1, nu1 = solvestar(P1)
    R2, M2, nu2 = solvestar(P2)
    Ps, Ms, Rs, nus = [P1, P2], [M1, M2], [R1, R2], [nu1, nu2]

    if visual:
        plt.ion() # automatically update open figure
        if stability:
            graph, = plt.plot([], [], "k-", zorder=0) # modify graph data later
            scatt = plt.scatter([], [], zorder=1) # modify graph data later
            cbar = plt.colorbar()
        else:
            graph, = plt.plot([], [], "k-o") # modify graph data later

    i = 0
    while i < len(Ps) - 1:
        P1, M1, R1, nu1 = Ps[i],   Ms[i],   Rs[i],   nus[i]
        P2, M2, R2, nu2 = Ps[i+1], Ms[i+1], Rs[i+1], nus[i+1]

        # Split intervals based on Euclidean distance between (R, M)-points in plot
        # But make sure P1, P2 do not get too close, otherwise algorithm gets stuck
        D = np.sqrt((R1 - R2)**2 + (M1 - M2)**2)
        if D > tolD and P2 - P1 > tolP:
            # split [P1, P2] into [P1, (P1+P2)/2] and [(P1+P2)/2, P2]
            P3 = (P1 + P2) / 2
            R3, M3, nu3 = solvestar(P3)
            Ps.insert(i+1, P3)
            Ms.insert(i+1, M3)
            Rs.insert(i+1, R3)
            nus.insert(i+1, nu3)

            if visual:
                # Animate plot in real-time for immediate feedback
                # inspired by https://stackoverflow.com/a/10944967
                graph.set_data(Rs, Ms)
                if stability:
                    scatt.set_offsets(np.transpose([Rs, Ms]))
                    scatt.set_array(np.array(nus))
                    scatt.set_cmap("jet")
                    scatt.set_clim(0, np.max(nus))
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
        utils.writecols([Ps, Ms, Rs, nus], ["P", "M", "R", "nu"], outfile)
    
    return Ps, Ms, Rs
