#!/usr/bin/sage

σ = var("σ", latex_name=r"\sigma")
σ0 = var("σ0", latex_name=r"\sigma_0")
σ1 = var("σ1", latex_name=r"\sigma_1")
π = var("π", latex_name=r"\pi")
π0 = var("π0", latex_name=r"\pi_0")
π1 = var("π1", latex_name=r"\pi_1")
m = var("m")
λ = var("λ", latex_name=r"\lambda")
h = var("h")

V(σ,π) = -1/2*m^2*(σ^2+π^2) + λ/24*(σ^2+π^2)^2 - h*σ

π0 = 0
σ0 = solve(diff(V(σ,π0), σ) == 0, σ)[1].rhs()

print(f"σ0 = {σ0}")
print(f"π0 = {π0}")

V(σ,π) = V(σ0+σ,π0+π).expand().maxima_methods().collectterms(σ,π)
