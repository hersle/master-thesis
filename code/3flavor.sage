#!/usr/bin/sage

# Express in terms of 0, 8
#σx, σy = var("σx", latex_name=r"\sigma_x"), var("σy", latex_name=r"\sigma_y")
#hx, hy = var("hx", latex_name=r"h_x"), var("hy", latex_name=r"h_y")
#transf = matrix([[sqrt(2/3), sqrt(1/3)], [sqrt(1/3), -sqrt(2/3)]])
#σx, σy = transf * vector([σ0, σ8])
#h0, h8 = transf.inverse() * vector([hx, hy])
#σ0, σ8 = transf.inverse() * vector([σx, σy])

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
σ0, σ1, σ2, σ3, σ4, σ5, σ6, σ7, σ8 = σ # unpack
h0, h1, h2, h3, h4, h5, h6, h7, h8 = h # unpack

σx, σy = var("σx", latex_name=r"\sigma_x"), var("σy", latex_name=r"\sigma_y")
transf = matrix([[sqrt(2/3), sqrt(1/3)], [sqrt(1/3), -sqrt(2/3)]])
σ08eqs = [
    σ0 == (transf.inverse() * vector([σx, σy]))[0],
    σ8 == (transf.inverse() * vector([σx, σy]))[1],
]

# verify correctness of definition
for i in range(0, 9):
    for j in range(0, 9):
        if i == j:
            assert (λ[i]*λ[j]).trace() == (2 if i==j else 0)

#def  comm(a, b): return a*b - b*a
#def acomm(a, b): return a*b + b*a
#def F(a, b, c): return 2/I * (comm(T[a],T[b])*T[c]).trace()
#def D(a, b, c): return 2 * (acomm(T[a],T[b])*T[c]).trace()
#F = [[[2/I * (comm(T[a],T[b])*T[c]).trace() for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]
#D = [[[2 * (acomm(T[a],T[b])*T[c]).trace() for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]
# TODO: these are ok, since T is hermitean?
tr2 = [[(T[a]*T[b]).trace() for b in range(0, 9)] for a in range(0, 9)]
tr4 = [[[[(T[a]*T[b]*T[c]*T[d]).trace() for d in range(0, 9)] for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]

trϕϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        trϕϕ += conjugate(ϕ[a]) * ϕ[b] * tr2[a][b]
trϕϕ = trϕϕ.simplify().expand()
#print(f"tr[ϕ†ϕ] = {trϕϕ}")

trϕϕϕϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        for c in range(0, 9):
            for d in range(0, 9):
                trϕϕϕϕ += conjugate(ϕ[a]) * ϕ[b] * conjugate(ϕ[c]) * ϕ[d] * tr4[a][b][c][d]
                #trϕϕϕϕ += σ[a]*σ[b]*σ[c]*σ[d] * tr4[a][b][c][d]
trϕϕϕϕ = trϕϕϕϕ.simplify().expand()
#print(f"tr[(ϕ†ϕ)²] = {trϕϕϕϕ}")

trHϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        trHϕ += h[a] * (conjugate(ϕ[b]) + ϕ[b]) * tr2[a][b]
        #trHϕ += 2*h[a]*σ[a]*tr2[a][b]
trHϕ = trHϕ.simplify().expand()
#print(f"tr[H(ϕ+ϕ†)] = {trHϕ}")

V = m2 * trϕϕ + λ1 * trϕϕ^2 + λ2 * trϕϕϕϕ - trHϕ
V = V.simplify().expand()

avgeqs = [σ[a]==0 for a in range(1, 8)] + [π[a]==0 for a in range(0, 9)]

# mass matrices
m2σσ = matrix(9, 9, lambda a, b: diff(V, σ[a], σ[b]).substitute(avgeqs).collect(λ1).collect(λ2))
m2σπ = matrix(9, 9, lambda a, b: diff(V, σ[a], π[b]).substitute(avgeqs).collect(λ1).collect(λ2))
m2ππ = matrix(9, 9, lambda a, b: diff(V, π[a], π[b]).substitute(avgeqs).collect(λ1).collect(λ2)) # TODO: one term in m2ππ[0,8] differs from Berge!

# find eigenvalues with given multiplicities eig[2] (3 for π, 4 for k)
m2σ = [eig[0] for eig in m2σσ.eigenvectors_left() if eig[2]==1][0].collect(λ1).collect(λ2) # TODO: there are two! which one to choose?
m2π = [eig[0] for eig in m2ππ.eigenvectors_left() if eig[2]==3][0].collect(λ1).collect(λ2)
m2K = [eig[0] for eig in m2ππ.eigenvectors_left() if eig[2]==4][0].collect(λ1).collect(λ2)

fπ = 93
fK = 113
numvals = [
    σ0 == ((fπ + 2*fK) / sqrt(6)).n(),
    σ8 == 2 * ((fπ - fK) / sqrt(3)).n(),
    m2σ == 800**2,
    m2π == 138**2,
    m2K == 496**2
]

import numpy as np
import scipy.optimize
def fun(x):
    funσ = (m2σ.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2])-800**2).n()
    funπ = (m2π.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2])-138**2).n()
    funK = (m2K.substitute(numvals, m2=x[0], λ1=x[1], λ2=x[2])-496**2).n()
    return np.array([funσ, funπ, funK])
sol = scipy.optimize.root(fun, (0, 0, 0))
print(f"m2 = -{sqrt(-sol.x[0])}^2, λ1 = {sol.x[1]}, λ2 = {sol.x[2]}")
# same numerical results as Berge!
