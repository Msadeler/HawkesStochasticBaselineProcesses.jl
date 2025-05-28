module HawkesStochasticBaselineProcesses

# Write your package code here.
using  Distributions
using  DataFrames
using Integrals
using  Optim
using Polynomials
using LinearAlgebra
using CairoMakie
using Statistics
import Base.rand
using HypothesisTests


const Parameters = Vector{Float64}
const BaselineParameters = Vector{Vector{Float64}}
const SubBaselineParameters = Union{Float64,Vector{Float64}}

abstract type MultiDimCov end
abstract type UniDimCov end
abstract type MultinomialProcess end
abstract type UninomialProcess end

const symboltypecov = Dict(
    :MDC=> MultiDimCov,
    :UDC => UniDimCov
)

const symboltypeprocess = Dict(
    :MP=> MultinomialProcess,
    :UP => UninomialProcess
)



export plot
export likelihood, gradient
export mle
export nbparams,params, params!, data!, gᵢX!,∫gᵢX! , nbdim
export UnivariatePolynomialFamilyBaseline, LinearFamilyBaseline,hessian,baseline, AbstractFamilyBaseline
export fisher
export  OneSampleTTest, EqualCoeffTest
export HawkesStochasticBaseline
export compensator
export procedureGOF, UniTest, ExpTest
export Baseline
export  MultiDimCov,UniDimCov, MultinomialProcess,UninomialProcess

include("family.jl")
include("model.jl")
include("Simulate.jl")
include("plots.jl")
include("likelihood.jl")
include("mle.jl")
include("fisherMatrix.jl")
include("test.jl")
include("compensator.jl")
include("GOF.jl")

end