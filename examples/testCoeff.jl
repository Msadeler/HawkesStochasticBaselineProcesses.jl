using HawkesStochasticBaselineProcesses
using  DataFrames
using LinearAlgebra
using Statistics
using Random
using Distributions
using CairoMakie

### Model declaration
g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


### Simulation
model = HawkesStochasticBaseline(0.6, 1.0, [1,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
T = 5000.0
df = rand(model, T)




### Estimation
model  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(model; data=df)

### Test of the null hypothesis H₁ : a ≠ 0.6 
OneSampleTTest(model, 0.6, 1)


### Test of the null hypothesis H₁ : m₁ = 0.2 
OneSampleTTest(model, 0.2, 3)

### Test of the null hypothesis H₁ : m₁ = m₂ 
EqualCoeffTest(model, 3,4)
