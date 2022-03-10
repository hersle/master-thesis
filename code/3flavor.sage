#!/usr/bin/sage

λ0 = matrix([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*sqrt(2/3)
λ1 = matrix([[0, 1, 0],[1, 0, 0],[0, 0, 0]])
λ2 = matrix([[0,-I, 0],[I, 0, 0],[0, 0, 0]])
λ3 = matrix([[1, 0, 0],[0,-1, 0],[0, 0, 0]])
λ4 = matrix([[0, 0, 1],[0, 0, 0],[1, 0, 0]])
λ5 = matrix([[0, 0,-I],[0, 0, 0],[I, 0, 0]])
λ6 = matrix([[0, 0, 0],[0, 0, 1],[0, 1, 0]])
λ7 = matrix([[0, 0, 0],[0, 0,-I],[0, I, 0]])
λ8 = matrix([[1, 0, 0],[0, 1, 0],[0, 0,-2]])*sqrt(1/3)
λ  = [λ0, λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8]
T  = [λ/2 for λ in λ]

m2 = var("m2", latex_name=r"m^2")
g = var("g")
λ1, λ2 = [var(f"λ{i}", latex_name=f"\\lambda_{i}") for i in range(1, 3)] # unpack
σ = [var(f"σ{a}", latex_name=f"\\sigma_{a}") for a in range(0, 9)]
π = [var(f"π{a}", latex_name=f"\\pi_{a}") for a in range(0, 9)]
h = [var(f"h{a}", latex_name=f"h_{a}") for a in range(0, 9)]
ϕ = [σ[a] + I*π[a] for a in range(0, 9)]

σx, σy = var("σx", latex_name=r"\sigma_x"), var("σy", latex_name=r"\sigma_y")
hx, hy = var("hx", latex_name=r"h_x"), var("hy", latex_name=r"h_y")
transf = matrix([[sqrt(2/3), sqrt(1/3)], [sqrt(1/3), -sqrt(2/3)]])
xyeqs = [
    σ[0] == (transf.inverse() * vector([σx, σy]))[0],
    σ[8] == (transf.inverse() * vector([σx, σy]))[1],
    h[0] == (transf.inverse() * vector([hx, hy]))[0],
    h[8] == (transf.inverse() * vector([hx, hy]))[1],
]
eqs08 = [
    σx == (transf * vector([σ[0], σ[8]]))[0],
    σy == (transf * vector([σ[0], σ[8]]))[1],
    hx == (transf * vector([h[0], h[8]]))[0],
    hy == (transf * vector([h[0], h[8]]))[1],
]
avgeqs = [σ[a]==0 for a in range(1, 8)] + [π[a]==0 for a in range(0, 9)]

# Compute potential symbolically with brute force
trϕϕ, trϕϕϕϕ, trHϕ = 0, 0, 0
for a in range(0, 9):
    for b in range(0, 9):
        trϕϕ += conjugate(ϕ[a])*ϕ[b] * (T[a]*T[b]).trace()
        trHϕ += h[a] * (conjugate(ϕ[b]) + ϕ[b]) * (T[a]*T[b]).trace()
        for c in range(0, 9):
            for d in range(0, 9):
                trϕϕϕϕ += conjugate(ϕ[a]*ϕ[c])*ϕ[b]*ϕ[d] * (T[a]*T[b]*T[c]*T[d]).trace()
V = expand(simplify(m2 * trϕϕ + λ1 * trϕϕ^2 + λ2 * trϕϕϕϕ - trHϕ))
Vtree = V.substitute(avgeqs, xyeqs).expand().simplify().collect(λ1).collect(λ2)
print(f"V = {Vtree}")

# mass matrices
m2σσ = matrix(9, 9, lambda a, b: diff(V,σ[a],σ[b]).substitute(avgeqs).collect(λ1).collect(λ2))
m2σπ = matrix(9, 9, lambda a, b: diff(V,σ[a],π[b]).substitute(avgeqs).collect(λ1).collect(λ2))
m2ππ = matrix(9, 9, lambda a, b: diff(V,π[a],π[b]).substitute(avgeqs).collect(λ1).collect(λ2)) # TODO: one term in m2ππ[0,8] differs from Berge!
for a in range(0, 9):
    for b in range(0, 9):
        print(f"m2σσ[{a},{b}] = {m2σσ[a,b]}")
for a in range(0, 9):
    for b in range(0, 9):
        print(f"m2σπ[{a},{b}] = {m2σπ[a,b]}")
for a in range(0, 9):
    for b in range(0, 9):
        print(f"m2ππ[{a},{b}] = {m2ππ[a,b]}")

# find eigenvalues with given multiplicities eig[2] (3 for π, 4 for k)
m2σ = m2σσ.eigenvalues()[0].collect(λ1).collect(λ2)
m2f = m2σσ.eigenvalues()[1].collect(λ1).collect(λ2)
m2a = m2σσ.eigenvalues()[2].collect(λ1).collect(λ2)
m2κ = m2σσ.eigenvalues()[5].collect(λ1).collect(λ2)
m2ηp = m2ππ.eigenvalues()[1].collect(λ1).collect(λ2)
m2η = m2ππ.eigenvalues()[0].collect(λ1).collect(λ2)
m2π = m2ππ.eigenvalues()[2].collect(λ1).collect(λ2)
m2K = m2ππ.eigenvalues()[5].collect(λ1).collect(λ2)
print(f"m2σ = {m2σ}\nm2π = {m2π}\nm2K = {m2K}")

eqhx = diff(V.substitute(xyeqs), σx).substitute(avgeqs).expand().simplify() == 0
eqhy = diff(V.substitute(xyeqs), σy).substitute(avgeqs).expand().simplify() == 0
solhxhy = solve([eqhx, eqhy], [hx, hy])
hx, hy = solhxhy[0][0].rhs(), solhxhy[0][1].rhs()
print(f"hx = {hx}\nhy = {hy}")

fπ = var("fπ", latex_name=r"f_\pi")
fK = var("fK", latex_name=r"f_K")
fπnum = 93 # TODO: 92.4?
fKnum = 113
m2σnum = 800^2
m2πnum = 138^2
m2Knum = 496^2
numvals = [
    σ[0] == ((fπnum + 2*fKnum) / sqrt(6)).n(),
    σ[8] == 2 * ((fπnum - fKnum) / sqrt(3)).n(),
]

import numpy as np
import scipy.optimize
def fun(x):
    funσ = (m2σ.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2]) - m2σnum).n()
    funπ = (m2π.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2]) - m2πnum).n()
    funK = (m2K.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2]) - m2Knum).n()
    return np.array([funσ, funπ, funK])
sol = scipy.optimize.root(fun, (-100, -10, +100), method="hybr")
m2num, λ1num, λ2num = sol.x[0], sol.x[1], sol.x[2]
numvals += [
    m2 == m2num,
    λ1 == λ1num,
    λ2 == λ2num,
]
hxnum = hx.substitute(eqs08).substitute(numvals).n()
hynum = hy.substitute(eqs08).substitute(numvals).n()

mud0 = 300 # up/down quark mass in vacuum
g = (2 * mud0 / fπnum).n()
msnum = (g*σy/sqrt(2)).substitute(eqs08).substitute(numvals).n()
print(f"ms = {msnum}")

print(f"m2 = -{sqrt(-m2num)}^2")
print(f"λ1 = {λ1num}")
print(f"λ2 = {λ2num}")
print(f"hx = {hxnum**(1/3)}^3")
print(f"hy = {hynum**(1/3)}^3")
print(f"g  = {g}")

Λx = (g*σx/(sqrt(e)*2)).substitute(eqs08).substitute(numvals).n()
Λy = (g*σy/(sqrt(e*2))).substitute(eqs08).substitute(numvals).n()
Λ  = ((2*Λx + Λy) / 3).n()
print(f"Λ = {Λ}")

# TODO: try to match parameters with Schaefer and Wagner
#θσ = (arctan(2*m2σσ[0,8]/(m2σσ[8,8]-m2σσ[0,0])) / 2).substitute(numvals).n()
#θπ = (arctan(2*m2ππ[0,8]/(m2ππ[8,8]-m2ππ[0,0])) / 2).substitute(numvals).n()
#m2fnum  = (m2σσ[0,0]*sin(θσ)^2 + m2σσ[8,8]*cos(θσ)^2 + m2σσ[0,8]*sin(2*θσ)).substitute(numvals).n()
#m2σnum  = (m2σσ[0,0]*sin(θσ)^2 + m2σσ[8,8]*cos(θσ)^2 - m2σσ[0,8]*sin(2*θσ)).substitute(numvals).n()
#m2anum  = (m2σσ[1,1]).substitute(numvals).n()
#m2κnum  = (m2σσ[4,4]).substitute(numvals).n()
#m2ηnum  = (m2ππ[0,0]*sin(θπ)^2 + m2ππ[8,8]*cos(θπ)^2 + m2ππ[0,8]*sin(2*θπ)).substitute(numvals).n()
#m2ηpnum = (m2ππ[0,0]*sin(θπ)^2 + m2ππ[8,8]*cos(θπ)^2 - m2ππ[0,8]*sin(2*θπ)).substitute(numvals).n()
#m2πnum  = (m2ππ[1,1]).substitute(numvals).n()
#m2Knum  = (m2ππ[4,4]).substitute(numvals).n()
m2σnum  = m2σ.substitute(numvals).n()
m2fnum  = m2f.substitute(numvals).n()
m2anum  = m2a.substitute(numvals).n()
m2κnum  = m2κ.substitute(numvals).n()
m2ηpnum = m2ηp.substitute(numvals).n()
m2ηnum  = m2η.substitute(numvals).n()
m2πnum  = m2π.substitute(numvals).n()
m2Knum  = m2K.substitute(numvals).n()
print(f"θσ = {θσ}")
print(f"θπ = {θπ}")
print(f"m2f  = {sqrt(m2fnum)}^2")
print(f"m2σ  = {sqrt(m2σnum)}^2")
print(f"m2a  = {sqrt(m2anum)}^2")
print(f"m2κ  = {sqrt(m2κnum)}^2")
print(f"m2η  = {sqrt(m2ηnum)}^2")
print(f"m2ηp = {sqrt(m2ηpnum)}^2")
print(f"m2π  = {sqrt(m2πnum)}^2")
print(f"m2K  = {sqrt(m2Knum)}^2")
#m2f = m2σσ[0,0]*sin(θσ)^2+m2σσ[8,8]*cos(θσ)
