using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Distributions

method = LBFGS()

Optim.order(method)
### Here gₘ(x) = m₁ + (m₂-m₁)exp(-‖x - x*‖² / 2)    


g₁(x)=1-exp(-norm(x-[0.2,0.2])^2/2) 
g₂(x)= exp(-norm(x-[0.2,0.2])^2/2)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x



### Definition and simulation of the model
model = HawkesStochasticBaseline(0.6, 1.0, [1 ,0.2];Mmax= 10, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df = rand(model,2000.0)
