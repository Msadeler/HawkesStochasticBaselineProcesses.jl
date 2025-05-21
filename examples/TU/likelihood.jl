using HawkesStochasticBaselineProcesses
using DataFrames
using Test


df = DataFrame( :time => 0:5, :timestamps => [false; fill(true, 4);false], :cov =>fill(1,6))
f(x)=1
gₘ = LinearFamilyBaseline([f])
model  = HawkesStochasticBaseline(0.0, 1.0,1.0;gₘ = gₘ)
l = likelihood(model, [0.6, 1.0,1.0], df)


outputPy = 6.3073

@test -likelihood(model, [0.6, 1.0,1.0], df) ≈ outputPy atol=10^(-1)