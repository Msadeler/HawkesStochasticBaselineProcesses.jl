using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics


g₁(x)=x^2
g₂(x)= 1
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df = rand(model, 3000.0)


################################
######## différences finies ####
################################

θ = [0.6 , 1.0, 1.0, 1.0]


ϵ = 0.0001
δϵ = zeros(4)
δϵ[4] = ϵ

likelihood(model,θ+δϵ, df)-likelihood(model,θ, df)- dot(gradient(model, θ,  df ), δϵ)


################################
######## Optim with grad #######
################################


modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelBGFS; data=df, method = BFGS())

params(modelBGFS)


modelNM  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelNM; data=df)
params(modelNM)
