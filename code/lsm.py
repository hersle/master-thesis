#!/usr/bin/python3

import utils

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

Nc = 3
Nf = 2
mσ = 550
mπ = 138
mq0 = 300
fπ = 93

g = mq0 / fπ
h = fπ * mπ**2
m = np.sqrt(1/2*(mσ**2-3*h/fπ))
λ = 6/fπ**3 * (h+m**2*fπ)

print(f"g = {g}")
print(f"h = {h} MeV^3")
print(f"m = {m} MeV")
print(f"λ = {λ}")

# Λ = mq0 / np.sqrt(np.e)

def localmini(x, tol=0):
    i = scipy.signal.argrelmin(x)[0]
    if len(i) > 0:
        return i[0]
    else:
        return None

    return np.argmin(np.abs(np.gradient(x)))
    for i in range(1, len(x)-1):
        if x[i] - x[i-1] <= tol and x[i] - x[i+1] <= tol:
            return i
    return int(len(x)/2)
    return None

def mq(σ): return g * np.abs(σ)
def ω0(σ): return -1/2*m**2*σ**2 + λ/24*σ**4
def ωr(σ): return Nc*Nf * mq(σ)**4 / (16*np.pi**2) * (3/2 + np.log(Λ**2/mq(σ)**2))
def ωf(σ,μ):
    if mq(σ) > 1: #and μ**2 - mq(σ)**2 > 0:
        x = 0 if μ**2-mq(σ)**2 < 0 else np.sqrt(μ**2 - mq(σ)**2) / mq(σ) # arg in sqrt(arg) should always be positive; if it is not then it is only slightly negative
        return -Nc * mq(σ)**4 / (24*np.pi**2) * ((2*x**3-3*x)*np.sqrt(x**2+1) + 3*np.arcsinh(x))
    else:
        return -1/12*Nc*μ**4/np.pi**2

minσ, minω, minμ = [], [], []
μs = np.arange(250, 1100, 25)
σs = np.linspace(-200, +200, 100)
rows = [[], [], []]
for μ in μs:
    #σ = np.linspace(-μ/g, μ/g, 1000)
    ωs = np.array([ω0(σ) + ωr(σ) + ωf(σ,μ) for σ in σs])
    ωs = ωs / fπ**4
    plt.plot(σs, ωs, "-k")

    for i in range(0, len(ωs)):
        rows[0].append(σs[i])
        rows[1].append(μ)
        rows[2].append(ωs[i])
    rows[0].append("")
    rows[1].append("")
    rows[2].append("")

    mini = localmini(ωs)
    if mini is not None:
        minσ.append(σs[mini])
        minω.append(ωs[mini])
        minμ.append(μ)
    else:
        minσ.append(0)
        minω.append(np.min(ωs))
        minμ.append(μ)

utils.writecols(rows, ["sigma", "mu", "omega"], "data/lsmpot.dat")

minσ = np.array(minσ)
minω = np.array(minω)
minμ = np.array(minμ)

utils.writecols([minσ, minμ, minω], ["sigma", "mu", "omega"], "data/lsmpotmin.dat")

plt.plot(minσ, minω, "r-o")

i = np.argmax(np.diff(minσ[1:] - minσ[:-1])) + 1
print(f"phase transition between μ = [{minμ[i]}, {minμ[i+1]}]")
plt.scatter((minσ[i]+minσ[i+1])/2, (minω[i]+minω[i+1])/2, marker="x", s=100, zorder=+100)

plt.xlim(σs[0], σs[-1])
plt.show()
