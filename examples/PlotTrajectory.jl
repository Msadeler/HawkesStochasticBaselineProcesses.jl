using CairoMakie
using HawkesStochasticBaselineProcesses
using LinearAlgebra

####################################################
#################### 1D Plot #######################
####################################################


g₁(x)=1
g₂(x)= abs(x)/200
coeff = [ g₂]
gₘ = LinearFamilyBaseline(coeff)

### dXₜ = -b(a-Xₜ)dt + σdWₜ 


function diffusion(x,t)
    z  = t/100%1
    return(30*(z-1/3)*(z<=1/3) + 10*(3*z-2)*(z>=2/3))
end

function drift(x,t)
    z =t/100%1
    return(0.05+40*(z-0.5)^2)
end


hsb = HawkesStochasticBaseline(0.3, 1.0, 0.5;Mmax= 200, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )
df = rand(hsb, 200.0)
HawkesStochasticBaselineProcesses.plot(hsb, :T)

####################################################
#################### 2D Plot #######################
####################################################


g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### dXₜ = -b(a-Xₜ)dt + σdWₜ 


drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

df = rand(model,200.0)

HawkesStochasticBaselineProcesses.plot(model,:T)


####################################################
############# 2D Plot along one axis ###############
####################################################


g₁(x)=1
g₂(x)= abs(x[1])/2
coeff = [ g₂]
gₘ = LinearFamilyBaseline(coeff)

### dXₜ = -b(a-Xₜ)dt + σdWₜ 


function diffusion(x,t)
    return([x[2], -6.5*x[2]+x[1]-0.6*x[1]^3 ])
end

function drift(x,t)
    return([0,2])
end

