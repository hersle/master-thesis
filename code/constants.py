import scipy.constants

# fundamental constants
π = scipy.constants.pi
ħ = scipy.constants.hbar
c = scipy.constants.c
mn = scipy.constants.neutron_mass

r0 = 1e3 # m
m0 = 1.98847e30 # kg, solar mass
b = 3
ϵ0 = m0*c**2 / (4*π*r0**3/b)
G = scipy.constants.G / (r0*c**2/m0) # dimensionless gravitational constant

fm = 1e-15
MeV = 1e6 * 1.6e-19
GeV = 1e9 * 1.6e-19
