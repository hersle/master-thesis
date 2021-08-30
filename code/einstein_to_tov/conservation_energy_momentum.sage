#!/usr/bin/sage

# inspiration: https://nbviewer.jupyter.org/github/sagemanifolds/SageManifolds/blob/master/Worksheets/v1.3/SM_TOV.ipynb

γ = var("γ", latex_name="\\gamma") # gravitational constant

M = Manifold(4, "M", structure="Lorentzian")

X.<t,r,θ,ϕ> = M.chart(r"t r:(0,+oo) θ:(0,pi):\theta ϕ:(0,2*pi):\phi")
α(r) = function("α", latex_name="\\alpha")(r)
m(r) = function("m")(r)
β(r) = function("β", latex_name="\\beta")(r)

g = M.metric(name="g")
g[0,0] = -exp(2*α(r))
g[1,1] = +exp(2*β(r))
g[2,2] = r^2
g[3,3] = r^2*sin(θ)^2

#G = g.ricci() - 1/2*g.ricci_scalar()*g
#G.set_name("G")

p(r) = function("p")(r)
ρ(r) = function("ρ", latex_name="\\rho")(r)
u = M.vector_field("u")
u[0] = exp(-α(r))
u = u.down(g)
u.set_name("u")

T = (ρ(r)+p(r)) * (u*u) + p(r) * g
T.set_name("T")

"""
eq1 = G[0,0].expr() == 8*pi*γ*T[0,0].expr()
eq2 = G[1,1].expr() == 8*pi*γ*T[1,1].expr()
eq3 = G[2,2].expr() == 8*pi*γ*T[2,2].expr()
eqs = [eq1, eq2, eq3]

eqm = eq1.solve(diff(m,r)(r))[0]
eqα = eq2.solve(diff(α,r)(r))[0]

dαdr(r) = eqα.rhs()

eqp = eq3.substitute(eqα).substitute(diff(α(r),r,r) == diff(eqα.rhs(),r)).solve(diff(p(r),r))[0].substitute(eqm)
dpdr(r) = eqp.rhs().factor()
"""

# display(dpdr(r)) # yay!
