using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim

hsb = HawkesStochasticBaseline(0.6, 1, Mmax= 1 )
timedata = rand(hsb,2000.0)


hsb_estim = HawkesStochasticBaseline(0.0,1.0, 1; baseline = (x,μ)-> μ)
## time change
θ = [1.0,0.0,1.0]
test = mle(hsb,θ, timedata)

params(hsb)
