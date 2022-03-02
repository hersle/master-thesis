#!/usr/bin/sage

σx, σy = var("σx", latex_name=r"\sigma_x"), var("σy", latex_name=r"\sigma_y")
hx, hy = var("hx", latex_name=r"h_x"), var("hy", latex_name=r"h_y")

xy = True
if xy:
    # Express in terms of 0, 8
    transf = matrix([[sqrt(2/3), sqrt(1/3)], [sqrt(1/3), -sqrt(2/3)]])
    h0, h8 = transf.inverse() * vector([hx, hy])
    σ0, σ8 = transf.inverse() * vector([σx, σy])
else:
    # Express in terms of x, y
    σ0, σ8 = var("σ0", latex_name=r"\sigma_0"), var("σ8", latex_name=r"\sigma_8")
    h0, h8 = var("h0", latex_name=r"h_0"), var("h8", latex_name=r"h_8")


λ0 = matrix([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*sqrt(2/3)
λ1 = matrix([[0, 1, 0],[1, 0, 0],[0, 0, 0]])
λ2 = matrix([[0,-I, 0],[I, 0, 0],[0, 0, 0]])
λ3 = matrix([[1, 0, 0],[0,-1, 0],[0, 0, 0]])
λ4 = matrix([[0, 0, 1],[0, 0, 0],[1, 0, 0]])
λ5 = matrix([[0, 0,-I],[0, 0, 0],[I, 0, 0]])
λ6 = matrix([[0, 0, 0],[0, 0, 1],[0, 1, 0]])
λ7 = matrix([[0, 0, 0],[0, 0,-I],[0, I, 0]])
λ8 = matrix([[1, 0, 0],[0, 1, 0],[0, 0,-2]])*sqrt(1/3)

σ  = [σ0, 0, 0, 0, 0, 0, 0, 0, σ8]
h  = [h0, 0, 0, 0, 0, 0, 0, 0, h8]
λ  = [λ0, λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8]
T  = [λ/2 for λ in λ]

# verify correctness of definition
for i in range(0, 9):
    for j in range(0, 9):
        if i == j:
            assert (λ[i]*λ[j]).trace() == 2
        else:
            assert (λ[i]*λ[j]).trace() == 0

def  comm(a, b): return a*b - b*a
def acomm(a, b): return a*b + b*a

#def F(a, b, c): return 2/I * (comm(T[a],T[b])*T[c]).trace()
#def D(a, b, c): return 2 * (acomm(T[a],T[b])*T[c]).trace()
#F = [[[2/I * (comm(T[a],T[b])*T[c]).trace() for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]
#D = [[[2 * (acomm(T[a],T[b])*T[c]).trace() for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]
tr2 = [[(T[a]*T[b]).trace() for b in range(0, 9)] for a in range(0, 9)]
tr4 = [[[[(T[a]*T[b]*T[c]*T[d]).trace() for d in range(0, 9)] for c in range(0, 9)] for b in range(0, 9)] for a in range(0, 9)]


trϕϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        trϕϕ += σ[a]*σ[b] * tr2[a][b]
trϕϕ = trϕϕ.simplify().expand()
print(f"tr[ϕ†ϕ] = {trϕϕ}")

trϕϕϕϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        for c in range(0, 9):
            for d in range(0, 9):
                trϕϕϕϕ += σ[a]*σ[b]*σ[c]*σ[d] * tr4[a][b][c][d]
trϕϕϕϕ = trϕϕϕϕ.simplify().expand()
print(f"tr[(ϕ†ϕ)²] = {trϕϕϕϕ}")

trHϕ = 0
for a in range(0, 9):
    for b in range(0, 9):
        trHϕ += 2*h[a]*σ[a]*tr2[a][b]
trHϕ = trHϕ.simplify().expand()
print(f"tr[H(ϕ+ϕ†)] = {trHϕ}")
