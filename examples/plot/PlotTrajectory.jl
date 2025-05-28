using HawkesStochasticBaselineProcesses
using LinearAlgebra
using CairoMakie

####################################################
#################### 1D Plot #######################
####################################################


gₘ = Baseline([LinearFamilyBaseline([x->abs(x)/200]),LinearFamilyBaseline([x-> 2-2*abs(x)/(abs(x)+1)])])

### dXₜ = -b(a-Xₜ)dt + σdWₜ 

function diffusion(x,t)
    z  = t/100%1
    return(30*(z-1/3)*(z<=1/3) + 10*(3*z-2)*(z>=2/3))
end

function drift(x,t)
    z =t/100%1
    return(0.05+10*(z-0.5)^2)
end

a =[0.6 -5 ;-5  0.6]
b =[2.0; 2.0]
m = [[1.0],[1.0]]


hsb = HawkesStochasticBaseline(a,b,m ;Mmax= 200.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )
df = rand(hsb, 500.0)

plot(hsb, :T)


####################################################
#################### 2D Plot #######################
####################################################

x¹ = [0.1,0.1]
x² = [-0.25, -0.25]
gₘ = Baseline( [LinearFamilyBaseline([ x-> 1-exp(-norm(x-x¹)*10), x-> exp(-norm(x-x¹)*10 )]), LinearFamilyBaseline([ x-> 1-exp(-norm(x-x²)*10), x-> exp(-norm(x-x²)*10 )])]   )

### dXₜ = -b(a-Xₜ)dt + σdWₜ 


drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x


a =[0.6 0.1 ;0.1  0.6]
b =[2.0; 2.0]
m = [[0.02, 1.0],[0.02, 1.0]]

model = HawkesStochasticBaseline(a,b,m;Mmax= 20.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

df = rand(model,5000.0)

plot(model,:T)

