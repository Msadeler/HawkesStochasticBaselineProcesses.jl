using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics


g₁(x)=norm(x)
g₂(x)= 1
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x,t)= 0.05
diffusion(x,t)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 20, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )
df = rand(model, 3000)

###############################


function gradient1(model::HawkesStochasticBaseline, θ::Vector, df::DataFrame)

    params!(model,θ)

    if isnothing(model.timedata)
        data!(model, df)
    end
    if isnothing(model.gᵢX)
        gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
    end
    if isnothing(model.∫gᵢX)
        ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )
    end    

    df[!,:gₘXₜ] = model.gᵢX*model.m

    gᵢXₜₖ=  model.gᵢX[df.timestamps,:]
    
    timeJump = df[df.timestamps, :]
    
    ∇λTₖ = [ 0 ;  0 ; gᵢXₜₖ[1,:]  ] ### values of ∇λTₖ
    ∇Λ = [  0 ; 0; model.∫gᵢX  ] ### compensator of ∇λ
    λTₖ = model.gₘ.(gᵢXₜₖ[1,:], model.m) ### values of λTₖ
    
    aux = model.a
    
    Tₖ₋₁,gₘTₖ₋₁ = timeJump[1,[:time, :gₘXₜ]]
    
    ∇l = ∇λTₖ/λTₖ
    
    
    for (i,jump) in enumerate(eachrow(timeJump[2:end, :]))
    
        δt = exp(-model.b*(jump.time - Tₖ₋₁))
    
        ### add ∫∂ₐλ between Tᵢ and Tᵢ₊₁ 
        ∇Λ[1]+= (1-δt)*(∇λTₖ[1]+1)/model.b 
    
        ### add ∫∂ᵦλ between Tᵢ and Tᵢ₊₁ 
        ∇Λ[2]+= 1/model.b *∇λTₖ[2]*(1- δt)- aux/model.b^2 *(1 - δt*( model.b*(jump.time - Tₖ₋₁) +1) )
        
        ### Update of ∇λ(Tᵢ₊₁)
        ∇λTₖ = [ δt *(∇λTₖ[1]+1); δt *( -(jump.time - Tₖ₋₁)*aux + ∇λTₖ[2] );gᵢXₜₖ[i+1,:] ]    
        ### Update of λ(Tᵢ₊₁)
        λTₖ = jump.gₘXₜ + δt*aux
    
    
        ## Update the gradient of the loglikelihood
        ∇l =∇l + ∇λTₖ/λTₖ
    
        ## Update of auxiliary variables
        Tₖ₋₁,gₘTₖ₋₁ = jump[[:time, :gₘXₜ]]
        aux = model.a + aux*δt
    end
    
    jump = df[end,:]
    
    δt = exp(-model.b*(jump.time - Tₖ₋₁))
    
    
    ∇Λ[1]+= (1-δt)*(∇λTₖ[1]+1)/model.b 
    ∇Λ[2]+= 1/model.b *∇λTₖ[2]*(1- δt)- aux/model.b^2 *(1 - δt*( model.b*(jump.time - Tₖ₋₁) +1))
    
    ∇l =∇l - ∇Λ
    
end





################################
######## différences finies ####
################################

θ = [0.6 , 1.0, 1.0, 1.0]


ϵ = 0.0001
δϵ = zeros(4)
δϵ[2] = ϵ

likelihood(model,θ+δϵ, df)-likelihood(model,θ, df)- dot(gradient(model, θ,  df ), δϵ)


################################
######## Optim with grad #######
################################


modelBGFS  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelBGFS; data=df, method = BFGS())
params(modelBGFS)


modelNM  = HawkesStochasticBaseline(0.0,1, [1.0,1.0], gₘ=gₘ)
mle(modelNM; data=df)
params(modelNM)
