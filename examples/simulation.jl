using HawkesStochasticBaselineProcesses



hsb = HawkesStochasticBaseline(0.6, 1, 100; baseline = x-> min(abs(x),2))


timedata = rand(hsb, 5)
plot(hsb, timedata,:IP)

theta = [2,0.6, 1]
### Likelihood
function baseline(x::Float64,mu::Float64)
    mu*abs(x)     
end



using  DataFrames
using Integrals

timedata = rand(hsb, 5)
