import scipy.constants

# fundamental constants
π = scipy.constants.pi
ħ = scipy.constants.hbar
c = scipy.constants.c
m = scipy.constants.neutron_mass

r0 = 1e4 # m
m0 = 1.98847e30 # kg, solar mass
G = scipy.constants.G / (r0*c**2/m0) # dimensionless gravitational constant
b = 3
ϵ0 = m0*c**2 / (4*π*r0**3/b)
