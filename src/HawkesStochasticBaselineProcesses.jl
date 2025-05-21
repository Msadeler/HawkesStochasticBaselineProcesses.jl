module HawkesStochasticBaselineProcesses

# Write your package code here.
using  Distributions
using  DataFrames
using Integrals
using  Optim
using Polynomials
using LinearAlgebra
using CairoMakie


import Base.rand



const Parameters = Vector{Float64}



export plot
export likelihood, gradient
export mle
export nbparams,params, params!, data!, gᵢX!,∫gᵢX! 
export UnivariatePolynomialFamilyBaseline, LinearFamilyBaseline,hessian,baseline, AbstractFamilyBaseline
export fisher

export HawkesStochasticBaseline, MultidimHawkesStochasticBaseline

include("family.jl")
include("model.jl")
include("simulateUnidim.jl")
include("plots.jl")
include("likelihood.jl")
include("mle.jl")
include("fisherMatrix.jl")
include("test.jl")



end