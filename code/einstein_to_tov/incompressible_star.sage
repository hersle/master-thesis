#!/usr/bin/sage

r = var("r")
e0 = var("e0", latex_name="e_0")
G = var("G")
p(r) = function("p")(r)
p0 = var("p0", latex_name="p_0")
m(r) = 4/3*pi*r^3*e0
R, M = var("R M")

# TODO: selecting ics2 gives same p(r) as in carroll, selecting ics1 gives p(r) in terms of p0
ics1 = [0, p0] # get things in terms of 
ics2 = [R, 0]
ics = ics2

diffeq = diff(p(r),r) == -(e0+p(r))*(G*m(r)+4*pi*G*r^3*p(r))/(r*(r-2*G*m(r)))
implsol = desolve(diffeq, ivar=r, dvar=p(r), ics=ics)
implsol = implsol.solve(p(r))[0]
eq = e^implsol.lhs() == e^implsol.rhs()
p(r) = eq.solve(p(r))[0].rhs()
p(r) = p(r).substitute(e0 = M / (4/3*pi*R^3))
p(r) = p(r).simplify_full().factor()
#sol = solve(eq, p(r))
#sol = solve(sol, p(r), algorithm="sympy")

eq1 = p(0) == p0
eq2 = M == 4/3*pi*R^3*e0

Rf(p0,e0) = sqrt(3/(4*pi)) * sqrt(p0/((e0+3*p0)*e0*G))
Mf(p0,e0) = 4/3*pi*Rf(p0,e0)^3*e0

f(x) = (p(x*R)/p(0)).substitute(M=3*R/(9*G)).substitute(R=2).real()
p1 = plot(f, (x, 0, 1))

"""
eq_M_R = p(0) == p0
Mf(R) = solve(eq_M_R, M)[0].rhs().factor()
Mf(R) = expand(numerator(Mf(R)) * I) / expand(denominator(Mf(R)) * I)
Mf(R) = Mf(R).factor()

#e0 = 1
eq = p0 == e0 * (R*sqrt(R-2*G*M)-R^(3/2)) / (R^(3/2)-3*R*sqrt(R-2*G*M))

e0 = 1
"""

# plot example working with ics2
#plt = plot(p(r).substitute(M=3*R/(9*G)).substitute(G=1).substitute(R=2).real(), r, 0, 2)
