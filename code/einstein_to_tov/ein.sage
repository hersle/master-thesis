#!/usr/bin/sage

# Inspired by https://nbviewer.jupyter.org/github/sagemanifolds/SageManifolds/blob/master/Worksheets/v1.3/SM_TOV.ipynb

γ = var("γ", latex_name="\\gamma") # gravitational constant
α(r) = function("α", latex_name="\\alpha")(r)
m(r) = function("m")(r)
β(r) = -1/2 * log(1 - 2*γ*m(r)/r)
p(r) = function("p")(r)
ρ(r) = function("ρ", latex_name="\\rho")(r)

M = Manifold(4, "M", structure="Lorentzian")
X.<t,r,θ,ϕ> = M.chart(r"t r:(0,+oo) θ:(0,pi):\theta ϕ:(0,2*pi):\phi")

g = M.metric(name="g")
g[0,0] = -exp(2*α(r))
g[1,1] = +exp(2*β(r))
g[2,2] = r^2
g[3,3] = r^2*sin(θ)^2

G = g.ricci() - 1/2*g.ricci_scalar()*g
G.set_name("G")

u = M.vector_field("u")
u[0] = exp(-α(r))
u = u.down(g)
u.set_name("u")

T = (ρ(r)+p(r)) * (u*u) + p(r) * g
T.set_name("T")

eq1 = G[0,0].expr() == 8*pi*γ*T[0,0].expr()
eq2 = G[1,1].expr() == 8*pi*γ*T[1,1].expr()
eq3 = G[2,2].expr() == 8*pi*γ*T[2,2].expr()

eqm = eq1.solve(diff(m,r)(r))[0]
eqα = eq2.solve(diff(α,r)(r))[0]

dαdr(r) = eqα.rhs()

eq3 = eq3.substitute(eqα, diff(α(r),r,r) == diff(dαdr(r),r))
eqp = eq3.solve(diff(p(r),r))[0]
eqp = eqp.substitute(eqm) # eliminate dm/dr term
dpdr(r) = eqp.rhs().factor()

print(dpdr(r))
