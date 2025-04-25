using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics
using Random

g₁(x)=1-exp(-norm(x-[0.02,0.02])^2/2) 
g₂(x)= exp(-norm(x-[0.02,0.02])^2/2)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 50, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

##########################################################


g₁(x)=abs(x)
g₂(x)= 1
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 50, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )

df = rand(model, 5000)

nrep =250
θhat = zeros(nrep,nbparams(model))
estimTime = 0
simuTime = 0

for k in 1:nrep

    start = time()
    df = rand(model, 5000)
    simuTime += time()- start

    timeBGFS = time()
    modelBGFS  = HawkesStochasticBaseline(0.0,1, [0.5,0.5], gₘ=gₘ)
    mle(modelBGFS; data=df, method=LBFGS())
    timeBGFS =  time()-timeBGFS 

    θhat[k,:] = params(modelBGFS)
    estimTime += timeBGFS

end

histogram(θhat[:,1])
mean(θhat[:,3])