using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics
using Random

g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )


### model
nrep =300
θhat = zeros(nrep,nbparams(model))
estimTime = 0
simuTime = 0



### simulation then estimation
for k in 1:nrep

    start = time()
    df = rand(model, 3000.0)
    simuTime += time()- start

    timeBGFS = time()
    modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
    mle(modelBGFS; data=df)
    timeBGFS =  time()-timeBGFS 

    θhat[k,:] = params(modelBGFS)
    estimTime += timeBGFS

end

histogram(θhat[:,2])
