using HawkesStochasticBaselineProcesses
using LinearAlgebra


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
    return(0.05+10*(z-0.5)^2)
end

a = [[0.6, 0.6 ] , [0.2, 0.4]]
b =  [[1, 1 ] , [2, 2]]


hsb = MultidimHawkesStochasticBaseline(a, b, 0.5;Mmax= 200, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )
df = rand(hsb, 500.0)
Ha