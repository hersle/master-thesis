#!/usr/bin/sage

m = var("m")
ϵ = var("ϵ", latex_name=r"\epsilon")
Λ = var("Λ", latex_name=r"\Lambda")

#t = taylor((e^euler_gamma*Λ/m)^(2*ϵ)*π^(-ϵ)*Γ(-2+ϵ), ϵ, 0, 0)
I = (4*π*Λ^2/(m^2))^ϵ*Γ(-2+ϵ)
t = taylor(I, ϵ, 0, 0)
