using HawkesStochasticBaselineProcesses
using LinearAlgebra


x¹ = [0.1,0.1]
x² = [-0.25, -0.25]
gₘ = Baseline( [LinearFamilyBaseline([ x-> 1-exp(-norm(x-x¹)*10), x-> exp(-norm(x-x¹)*10 )]), LinearFamilyBaseline([ x-> 1-exp(-norm(x-x²)*10), x-> exp(-norm(x-x²)*10 )])]   )

### dXₜ = -b(a-Xₜ)dt + σdWₜ 


drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


a =[0.6 0.2 ;0.3  0.6]
b =[2.0; 1.5]
m = [[0.2, 1.0],[0.5, 1.0]]

model = HawkesStochasticBaseline(a,b,m;Mmax= 20.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df = rand(model, 700.0)


timeBGFS = time()
modelBGFS  = HawkesStochasticBaseline(ones((2,2)),2 .* ones((2,1)), [[1.0, 1.0],[1.0, 1.0]], gₘ=gₘ)
mle(modelBGFS; data=df)
timeBGFS =  time()-timeBGFS 

