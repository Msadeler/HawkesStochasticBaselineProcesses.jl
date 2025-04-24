using HawkesStochasticBaselineProcesses
using  DataFrames
using Optim
using Plots
using Integrals
using LinearAlgebra
using Statistics

g₁(x)= 1.0
gₘ = LinearFamilyBaseline([g₁])


model = HawkesStochasticBaseline(0.6, 1.0, 1.0, gₘ = gₘ, Mmax=1.0)
df = DataFrame(:time=> 0:4, :timestamps=> [false, true, true,true,false], :cov => [1.0,1.0, 1.0, 1.0, 1.0])

θ = [0.6 , 1.0, 1.0]
gradient(model, θ, df)

df[!,:gₘXₜ] = model.gᵢX*model.m

gᵢXₜₖ=  model.gᵢX[df.timestamps,:]

timeJump = df[df.timestamps, :]

∇λTₖ = [ 0 ;  0 ; gᵢXₜₖ[1,:]  ] ### values of ∇λTₖ
∇Λ = [  0 ; 0; model.∫gᵢX  ] ### compensator of ∇λ
λTₖ = dot(gᵢXₜₖ[1,:], model.m) ### values of λTₖ

aux = model.a

Tₖ₋₁,gₘTₖ₋₁ = timeJump[1,[:time, :gₘXₜ]]

∇l = ∇λTₖ/λTₖ


for (i,jump) in enumerate(eachrow(timeJump[2:end, :]))

    print(aux, "\n" )
    δt = exp(-model.b*(jump.time - Tₖ₋₁))

    ### add ∫∂ₐλ between Tᵢ and Tᵢ₊₁ 
    ∇Λ[1]+= (1-δt)*(∇λTₖ[1]+1)/model.b 

    ### add ∫∂ᵦλ between Tᵢ and Tᵢ₊₁ 
    ∇Λ[2]+= 1/model.b *∇λTₖ[2]*(1- δt)- aux/model.b^2 *(1 - δt*( model.b*(jump.time - Tₖ₋₁) +1))
    
    ### Update of ∇λ(Tᵢ₊₁)
    ∇λTₖ = [ δt *(∇λTₖ[1]+1); δt *( -(jump.time - Tₖ₋₁)*aux + ∇λTₖ[2] );gᵢXₜₖ[i+1,:] ]    
    ### Update of λ(Tᵢ₊₁)
    λTₖ = jump.gₘXₜ + δt*aux


    ## Update the gradient of the loglikelihood
    ∇l =∇l + ∇λTₖ/λTₖ
    print(λTₖ, "\n", ∇λTₖ, "\n", ∇Λ, "\n")

    ## Update of auxiliary variables
    Tₖ₋₁,gₘTₖ₋₁ = jump[[:time, :gₘXₜ]]
    aux = model.a + aux*δt
end

jump = df[end,:]

δt = exp(-model.b*(jump.time - Tₖ₋₁))


∇Λ[1]+= (1-δt)*(∇λTₖ[1]+1)/model.b 
∇Λ[2]+= 1/model.b *∇λTₖ[2]*(1- δt)- aux/model.b^2 *(1 - δt*( model.b*(jump.time - Tₖ₋₁) +1))

∇l =∇l - ∇Λ

print(∇Λ[1])


####

############# différences finies 


df = rand(model, 5000.0)
ϵ = 0.000001
δϵ = zeros(3)
δϵ[2] = ϵ
θ = [0.6 , 1.0, 1.0]
likelihood(model,θ+δϵ, df)-likelihood(model,θ, df)- dot(gradient(model, θ,  df ), δϵ)
