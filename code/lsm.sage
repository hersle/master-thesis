#!/usr/bin/sage

Nc = var("Nc", latex_name=r"N_c")
σ = var("σ", latex_name=r"\sigma")
μ = var("μ", latex_name=r"\mu")
g = var("g")
mq = g^4 * σ^4
x = sqrt(μ^2 - mq^2) / mq

ω(σ,μ) = -Nc * mq**4 / (24*π**2) * ((2*x^3-3*x)*sqrt(x^2+1))# + 3*arcsinh(x))
t = taylor(ω(σ,μ), σ, 0, 2)
print(t)
