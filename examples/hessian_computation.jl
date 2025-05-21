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

model = HawkesStochasticBaseline(0.6, 1.0, [1.0,1.0];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df = rand(model,8000.0)


θ = [0.6,1,0.2,1]

if isnothing(model.timedata)
    data!(model, df)
end
if isnothing(model.gᵢX)
    gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
end
if isnothing(model.∫gᵢX)
    ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )
end  



function hessian(model::HawkesStochasticBaseline, df::DataFrame)
    
    data = copy(df)
    timeJump = data[data.timestamps, :]
    timeJump[:,:gₘXₜ] = [model.gₘ(x, model.m)  for x in timeJump.cov]

    gᵢXₜₖ=  model.gᵢX[data.timestamps,:]


    n = nbparams(model)


    ∇²l = zeros(n,n)
    ∇²λTₖ = zeros(n,n)
    ∇²Λ = zeros(n,n)

    ∇Λ = [0;0;zeros(length(model.m))]
    ∇λTₖ = [0;0;gᵢXₜₖ[1,:] ]
    λTₖ = timeJump.gₘXₜ[1]


    ∇²l  = (∇²λTₖ - ∇λTₖ*∇λTₖ')/λTₖ^2 - ∇²Λ

    aux = model.a
    stockVariable = ∇λTₖ[2]

    Tₖ₋₁,gₘTₖ₋₁ = timeJump[1,[:time, :gₘXₜ]]


    for (i,jump) in enumerate(eachrow(timeJump[2:end, :]))
            

        δt = jump.time - Tₖ₋₁

        ∇Λ[1]= (1-exp(model.b*δt))*(∇λTₖ[1]+1)
        ∇Λ[2]= aux/model.b^2*( 1- exp(-model.b*δt)*(model.b*δt+1) )  + ∇λTₖ[2]/model.b*(1-exp(-model.b*δt))

        ∇²Λ[1,2] = ∇Λ[2]/model.a
        ∇²Λ[1,2] = ∇²Λ[2,1]
        ∇²Λ[2,2] = ∇²λTₖ[2, 2] /model.b*(1- exp(-δt*model.b)) + 2*∇λTₖ[2]/model.b^2*(1- exp(-model.b*δt)*(model.b*δt+1)) + aux/model.b^3*(2- exp(-model.b*δt)*(model.b^2*δt^2 + 2*model.b*δt+2))

        ∇λTₖ = [exp(-model.b*δt)*(∇λTₖ[1]+ 1); exp(-model.b*δt)*(-δt*aux+∇λTₖ[2]);gᵢXₜₖ[i+1,:]]

        ∇²λTₖ[2, 2] = exp(-model.b*δt)*(∇²λTₖ[2, 2] -δt*stockVariable) -δt*∇λTₖ[2]
        ∇²λTₖ[1, 2]= ∇λTₖ[2]/model.a
        ∇²λTₖ[2, 1] = ∇²λTₖ[1, 2]

        λTₖ = jump.gₘXₜ + exp(-model.b*δt)*aux

        stockVariable =∇λTₖ[2]

        ∇²l = ∇²l +  (∇²λTₖ - ∇λTₖ*∇λTₖ')/λTₖ^2 - ∇²Λ
        Tₖ₋₁,gₘTₖ₋₁ = jump[[:time, :gₘXₜ]]
        aux = model.a + aux*δt
    end

    jump = data[end,:]

    δt = jump.time - Tₖ₋₁

    ∇Λ[2]= aux/model.b^2*( 1- exp(-model.b*δt)*(model.b*δt+1) )  + ∇λTₖ[2]/model.b*(1-exp(-model.b*δt))


    ∇²Λ[1,2] = ∇Λ[2]/model.a
    ∇²Λ[1,2] = ∇²Λ[2,1]
    ∇²Λ[2,2] = ∇²λTₖ[2, 2] /model.b*(1- exp(-δt*model.b)) + 2*∇λTₖ[2]/model.b^2*(1- exp(-model.b*δt)*(model.b*δt+1)) + aux/model.b^3*(2- exp(-model.b*δt)*(model.b^2*δt^2 + 2*model.b*δt+2))
    ∇²l = ∇²l - ∇²Λ

end

nrep =200
θhat = zeros(nrep,nbparams(model))
estimator = zeros(nrep, nbparams(model))
∇lhat = zeros(nrep,nbparams(model))  



T = 10000.0
k=1
df = rand(model, T)

modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelBGFS; data=df)
θhat[k,:] = params(modelBGFS)

model.gᵢX

df.timestamps

fisher(model, df)

∇lhat[k,:] = diag(sqrt(inv(fisher(model, df))))

estimator[k,:] = sqrt(T)*(θhat[k,:]- θ)./∇lhat[k,:]



for k in 1:nrep

    df = rand(model, T)

    modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
    mle(modelBGFS; data=df)
    θhat[k,:] = params(modelBGFS)

    ∇lhat[k,:] = diag(sqrt(inv(fisher(model, df))))

    estimator[k,:] = sqrt(T)*(θhat[k,:]- θ)./∇lhat[k,:]

   
end


histogram(estimator[:,4])

std(eachrow(estimator))

std(eachrow(θhat))

mean(eachrow(∇lhat))