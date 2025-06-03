using HawkesStochasticBaselineProcesses
using LinearAlgebra
using Random


x¹ = [0.1,0.1]
x² = [-0.25, -0.25]

g(x,m) = m[1] + (m[2]-m[1])*exp(-m[3]*norm(x-x¹) )


gₘ = Baseline([GeneralFamilyBaseline(g)])


Random.seed!(0)

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


a =0.6*ones(1,1)
b =[1.0]
m = [[0.02, 1.0,10]]

model = HawkesStochasticBaseline(a,b,m;Mmax= 20.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

df = rand(model, 10.0)


loglikelihood(model, params(model),df)