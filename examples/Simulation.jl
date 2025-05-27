using HawkesStochasticBaselineProcesses
using Distributions
using LinearAlgebra
using DataFrames

g1, g2 =  LinearFamilyBaseline([x->abs(x)]),LinearFamilyBaseline([x->1])
gₘ = Baseline([ [g1], [g2]])

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


a =[0.6 0.6; 0.6  0.6]
b =[2.0; 2.0]
m = [[1.0],[1.0]]
hsb = HawkesStochasticBaseline(a,b,m;Mmax= [1.0, 1.0], gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )

rand(hsb,20.0)
