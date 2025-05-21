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

T = 3000.0


nrep =150
θhat = zeros(nrep,nbparams(model))
estimator = zeros(nrep, nbparams(model))
∇lhat = zeros(nrep,nbparams(model), nbparams(model) )  

stat = zeros(nrep)

T = 2000.0
θ = [0.6,1,0.2,1]

for k in 1:nrep

    df = rand(model, T)

    modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ )
    mle(modelBGFS; data=df)
    θhat[k,:] = params(modelBGFS)

    ∇lhat[k,:] = sqrt(inv(fisher(modelBGFS, df)))

    estimator[k,:] = sqrt(T)*(θhat[k,:]- θ)./∇lhat[k,:]


   
end
