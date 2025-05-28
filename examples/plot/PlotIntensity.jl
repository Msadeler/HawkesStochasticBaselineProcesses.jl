using HawkesStochasticBaselineProcesses
using LinearAlgebra
using DataFrames

####################################################
#################### 1D Plot #######################
####################################################


gₘ = Baseline([[LinearFamilyBaseline([x->abs(x)/200])],[LinearFamilyBaseline([x-> 2-2*abs(x)/(abs(x)+1) ])]])

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


