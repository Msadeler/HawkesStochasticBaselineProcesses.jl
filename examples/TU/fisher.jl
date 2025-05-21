using HawkesStochasticBaselineProcesses
using DataFrames
using Test


df = DataFrame( :time => 0:5, :timestamps => [false; fill(true, 4);false], :cov =>fill(1,6))
f(x)=1
gₘ = LinearFamilyBaseline([f])


model  = HawkesStochasticBaseline(0.6, 1.0,1.0;gₘ = gₘ)
outputPy = reduce(hcat,[[  0.08252534,  -0.06312538, 0.17110561], [-0.06312538,    0.04905937,-0.12813741], [0.17110561, -0.12813741,  0.56496415]])

@test fisher(model, df) ≈ outputPy atol= 10^(-4)
