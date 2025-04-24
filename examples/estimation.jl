using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics
using Random
###################################################################################################################
#################################################  Xₜ multidim  ###################################################
###################################################################################################################

### Here gₘ(x) = m₁ + (m₂-m₁)exp(-‖x - x*‖² / 2)    

Random.seed!(0)

g₁(x)=1-exp(-norm(x-[0.2,0.2])^2/2) 
g₂(x)= exp(-norm(x-[0.2,0.2])^2/2)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df =rand(model, 2000.0)


###### First order method
timeNM = time()
 modelNM  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelNM; data=df)
timeNM =  time()-timeNM 

params(modelNM)

###### Second order method
timeBGFS = time()
modelBGFS  = HawkesStochasticBaseline(0.0,1, [0.5,0.5], gₘ=gₘ)
mle(modelBGFS; data=df, method=LBFGS())
timeBGFS =  time()-timeBGFS 
params(modelBGFS)



histogram(param[:,3])


mean(param[:,3])