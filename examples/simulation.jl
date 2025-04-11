using HawkesStochasticBaselineProcesses

### simulation of the process

hsb = HawkesStochasticBaseline(0.6, 1, 100; baseline = x-> min(abs(x),2))
timedata = rand(hsb, 5)

function baseline(x::Float64,mu::Float64)
    mu*abs(x)     
end


### Likelihood computation of the proceess

theta = [2,0.6, 1]

model = Model(timedata, baseline)

likelihood(theta,model)
