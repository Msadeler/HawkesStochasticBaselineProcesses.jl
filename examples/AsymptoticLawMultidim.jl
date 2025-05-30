using HawkesStochasticBaselineProcesses
using LinearAlgebra
using Random
using BenchmarkTools

x¹ = [0.1,0.1]
x² = [-0.25, -0.25]


gₘ = Baseline( [LinearFamilyBaseline([ x-> 1-exp(-norm(x-x¹)*10), x-> exp(-norm(x-x¹)*10 )]), 
    LinearFamilyBaseline([ x-> 1-exp(-norm(x-x²)*10), x-> exp(-norm(x-x²)*10 )])]   
    )



### dXₜ = -b(a-Xₜ)dt + σdWₜ 

Random.seed!(0)

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


a =[0.6 0.1 ;0.1  0.6]
b =[2.0; 2.0]
m = [[0.02, 1.0],[0.02, 1.0]]

model = HawkesStochasticBaseline(a,b,m;Mmax= 20.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

df = rand(model, 1000.0)

HawkesStochasticBaselineProcesses.prepareintegral!(model.gₘ, df)

@btime loglikelihood($model, params($model),$df)



nrep =1
estimTime = 0
simuTime = 0
θhat= zeros((nrep,length(model)))

for k in 1:nrep

    start = time()
    df = rand(model, 1000.0)
    simuTime += time()- start

    timeBGFS = time()
    modelBGFS  = HawkesStochasticBaseline(ones((2,2)),2 .* ones((2,1)), [[1.0, 1.0],[1.0, 1.0]], gₘ=gₘ)
    mle(modelBGFS; data=df)
    timeBGFS =  time()-timeBGFS 
    estimTime+=timeBGFS

    θhat[k,:]= params(modelBGFS)


end
