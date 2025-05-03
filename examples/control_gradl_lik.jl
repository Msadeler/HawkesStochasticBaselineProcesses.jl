using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics


g₁(x)=abs(x)
g₂(x)= 1
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 50, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )
df = rand(model, 2000)


################################
######## différences finies ####
################################

θ = [0.6 , 1.0, 1.0, 1.0]


ϵ = 0.01
δϵ = zeros(4)
δϵ[4] = ϵ

likelihood(model,θ+δϵ, df)-likelihood(model,θ, df)- dot(gradient(model, θ,  df ), δϵ)
