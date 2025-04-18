using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals

### Baseline

f(x)=1
g(x)=min(x^2,4)
coeff = [f; g]
gₘ = LinearFamilyBaseline(coeff)


### EDS coefficients

drift(x)= x
diffusion(x)=1

### simulation
model = HawkesStochasticBaseline(0.6, 1.0, [1 ,0.2];Mmax= 10, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=0.0 )
df = rand(model,5000)



#### Estimation
model = HawkesStochasticBaseline(1, 0.0, [1.0,1.0]; gₘ = gₘ)
θ = [1, 0.0, 1.0,1.0]


### gradient likelihood

if isnothing(model.data)
    print("No data has been providen")
end

model = HawkesStochasticBaseline(0, 1.0, [1.0 ,1.0];gₘ = gₘ)

data!(model, df)
gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )

##

timeJump = df[df.timestamps, :]
∇λTₖ = [ timeJump.∇gₘX ;  0 ;  0 ]
∇Λ = [model  ; 0 ; 0 ]
λTₖ = model.baseline(timeJump.time[1], model.m)
