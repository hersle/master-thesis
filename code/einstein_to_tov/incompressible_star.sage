#!/usr/bin/sage

r = var("r")
e0 = var("e0", latex_name="e_0")
G = var("G")
p0 = var("p0", latex_name="p_0")
m(r) = 4/3*pi*r^3*e0
R, M = var("R M")

# TODO: selecting ics2 gives same p(r) as in carroll, selecting ics1 gives p(r) in terms of p0
ics1 = [0, p0] # get things in terms of central pressure
ics2 = [R, 0]  # get things in terms of R
ics = ics2

def relativistic_pressure():
    p(r) = function("p")(r)
    diffeq = diff(p(r),r) == -(e0+p(r))*(G*m(r)+4*pi*G*r^3*p(r))/(r*(r-2*G*m(r)))
    implsol = desolve(diffeq, ivar=r, dvar=p(r), ics=ics)
    implsol = implsol.solve(p(r))[0]
    eq = e^implsol.lhs() == e^implsol.rhs()
    p(r) = eq.solve(p(r))[0].rhs()
    p(r) = p(r).substitute(e0 = M / (4/3*pi*R^3))
    p(r) = p(r).simplify_full().factor()
    return p(r)

def newtonian_pressure():
    p(r) = function("p")(r)
    diffeq = diff(p(r),r) == -G*m(r)*e0/r^2
    p(r) = desolve(diffeq, dvar=p(r), ivar=r, ics=ics)
    p(r) = p(r).substitute(e0^2 == e0 * M/(4/3*pi*R^3))
    p(r) = p(r).factor()
    return p(r)

def plot_pressure(p, M, R, **kwds):
    f(x) = (p(x*R)/p(0)).substitute(M=M).substitute(R=R).real()
    return plot(f, (x, 0, 1), **kwds)

p1(r) = newtonian_pressure()
print(f"Newtonian pressure: {p1(r)}")
plot1 = plot_pressure(p1, 7*R/(9*G), 2, color="red")

p2(r) = relativistic_pressure()
print(f"Relativistic pressure: {p2(r)}")
plot2 = plot_pressure(p2, 7*R/(9*G), 2, color="green")

print("Relativistic pressure for GM/R << 1 same as Newtonian pressure? ", end="")
print(bool((p2(r).substitute(G=R/M*x).simplify_full().taylor(x, 0, 1).substitute(x=G*M/R) - p1(r)).substitute(e0=M/(4*pi*R^3/3)).simplify_full() == 0))

plot12 = plot1 + plot2

Rf(p0,e0) = sqrt(3/(4*pi)) * sqrt(p0/((e0+3*p0)*e0*G))
Mf(p0,e0) = 4/3*pi*Rf(p0,e0)^3*e0

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
