module HawkesStochasticBaselineProcesses

# Write your package code here.
using  Distributions
using  DataFrames
using Plots
using Integrals
using  Optim



const Parameters = Vector{Float64}



export plot
export likelihood 
export mle
export nbparams,params, params!, data!, gᵢX!,∫gᵢX! 

include("family.jl")
include("model.jl")
include("simulate.jl")
include("plots.jl")
include("likelihood.jl")
include("mle.jl")


export UnivariatePolynomialFamilyBaseline, LinearFamilyBaseline,gradient,hessian,baseline, AbstractFamilyBaseline
export HawkesStochasticBaseline

end