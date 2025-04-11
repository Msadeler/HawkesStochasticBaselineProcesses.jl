module HawkesStochasticBaselineProcesses

# Write your package code here.
using  Distributions
using  DataFrames
using Plots
using Integrals


export plot


include("model.jl")
include("simulate.jl")
include("plots.jl")
include("likelihood.jl")

export HawkesStochasticBaseline


end