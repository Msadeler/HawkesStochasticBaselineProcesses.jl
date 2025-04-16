using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots

### Simulation
hsb = HawkesStochasticBaseline(0.6, 1, Mmax= 1 )
timedata = rand(hsb,5.0)


plot(hsb, timedata, :IP)

coeff = x -> x
baseline = HawkesStochasticBaselineProcesses.UnivariatePolynomialFamilyBaseline(coeff)

HawkesStochasticBaselineProcesses.gradient(1.0,2.0,baseline)

## Estimation
hsb_estim = HawkesStochasticBaseline(0.0,1.0, 1; baseline = (x,μ)-> μ)

θ = [1.0,0.0,1.0]
test = mle(hsb,θ, timedata)

params(hsb)
