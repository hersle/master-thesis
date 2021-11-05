#!/usr/bin/sage

r = var("R") # constant radius

M = Manifold(2, "M")
X.<θ,ϕ> = M.chart(r"θ:(0,pi):\theta ϕ:(0,2*pi):\phi")

g = M.metric(name="g")
g[0,0] = r^2
g[1,1] = r^2*sin(θ)^2
