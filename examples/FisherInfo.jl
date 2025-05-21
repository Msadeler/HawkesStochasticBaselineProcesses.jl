using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics
using Random



g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )

function hessianTest(model::HawkesStochasticBaseline, data::DataFrame)

    df = copy(data)

    if isnothing(model.timedata)
        data!(model, df)
    end
    if isnothing(model.gᵢX)
        gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
    end
    if isnothing(model.∫gᵢX)
        ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )
    end  
    
    timeJump = df[df.timestamps, :]
    timeJump[:,:gₘXₜ] = [model.gₘ(x, model.m)  for x in timeJump.cov]

    gᵢXₜₖ=  model.gᵢX[data.timestamps,:]
    ∇λTₖ = [0;0;gᵢXₜₖ[1,:] ]
    λTₖ = model.gₘ( timeJump.cov[1], model.m)

    Γ = ∇λTₖ*∇λTₖ' / λTₖ^2


    Tₖ₋₁, gₘXₜₖ₋₁ =  timeJump[1,[:time, :gₘXₜ]]


    for (k,jump) in enumerate(eachrow(timeJump[2:end, :]))


        ∇λTₖ[[1,2]] = exp(-model.b*(jump.time -  Tₖ₋₁))* (∇λTₖ[[1,2]] .+  [ 1 ; (λTₖ + model.a - jump.gₘXₜ)*(jump.time - Tₖ₋₁) ])  

        ∇λTₖ[3:end] = gᵢXₜₖ[k+1,:]

        λTₖ = jump.gₘXₜ +  exp(-model.b*(jump.time -  Tₖ₋₁))*( λTₖ + model.a - gₘXₜₖ₋₁ )

        Γ += ∇λTₖ*∇λTₖ' / λTₖ^2

        Tₖ₋₁ = jump.time
        gₘXₜₖ₋₁ =  jump.gₘXₜ


        
    end

    return(Γ)
    
    
end



nrep =150
θhat = zeros(nrep,nbparams(model))
estimator = zeros(nrep, nbparams(model))
∇lhat = zeros(nrep,nbparams(model))  



T = 2000.0
θ = [0.6,1,0.2,1]

for k in 1:nrep

    df = rand(model, T)

    modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
    mle(modelBGFS; data=df)
    θhat[k,:] = params(modelBGFS)

    ∇lhat[k,:] = diag(sqrt(inv(hessianTest(modelBGFS, df)/T)))

    estimator[k,:] = sqrt(T)*(θhat[k,:]- θ)./∇lhat[k,:]

   
end



x =estimator[:,4]
R"
library(qqconf)
qq_conf_plot($x, distribution = qnorm, 
           dparams = list('mean' =0,'sd'=1), 
           polygon_params = list( col='powderblue', border = FALSE),
           points_params = list(cex=0.2))"