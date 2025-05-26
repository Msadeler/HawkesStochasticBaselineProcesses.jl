using HawkesStochasticBaselineProcesses
using LinearAlgebra

g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)


g = Baseline([[gₘ]])

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [[0.2,1.0]];Mmax= 20, gₘ = g, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

g([0.0, 0.0], [[0.2,1]])

df = rand(model, 200.0)

model.m