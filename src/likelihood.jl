
function likelihood(model::HawkesStochasticBaseline, θ::Vector{Float64}, df::DataFrame)

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
    
    
    Tₖ₋₁,gₘTₖ₋₁ = df[df.timestamps,[:time, :gₘXₜ]][1,:]


    if sum(df.gₘXₜ.<0)>0 || model.b <= 0 || model.a <=0
        -1e30    
    else
        λTₖ = gₘTₖ₋₁
        ∑lnλTₖ  = log(λTₖ)

        Λ =  model.∫gᵢX'*model.m   ### integrate of baseline along the trajectory of the covariate


        for jump in eachrow(df[(df.timestamps) .& (df.time.>Tₖ₋₁),:])

            Λ += (1- exp(-model.b*(jump.time - Tₖ₋₁)))*(λTₖ + model.a - gₘTₖ₋₁)/model.b ### compute the compensator
            λTₖ = jump.gₘXₜ + exp(-model.b*(jump.time- Tₖ₋₁))*(λTₖ+ model.a - gₘTₖ₋₁) ## compute λ(Tk+1)
            ∑lnλTₖ+= log(λTₖ)

            Tₖ₋₁, gₘTₖ₋₁ = jump.time, jump.gₘXₜ 
            
            
        end

        jump = df[end,:]

        Λ += (1- exp(-model.b*(jump.time - Tₖ₋₁)))*(λTₖ + model.a - gₘTₖ₋₁)/model.b

        return(∑lnλTₖ-Λ)        
    end

end



function gradient(model::HawkesStochasticBaseline, θ::Vector, df::DataFrame)

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
    λTₖ = dot(gᵢXₜₖ[1,:], model.m) ### values of λTₖ
    
    aux = model.a
    
    Tₖ₋₁,gₘTₖ₋₁ = timeJump[1,[:time, :gₘXₜ]]
    
    ∇l = ∇λTₖ/λTₖ
    
    
    for (i,jump) in enumerate(eachrow(timeJump[2:end, :]))
    
        δt = exp(-model.b*(jump.time - Tₖ₋₁))
    
        ### add ∫∂ₐλ between Tᵢ and Tᵢ₊₁ 
        ∇Λ[1]+= (1-δt)*(∇λTₖ[1]+1)/model.b 
    
        ### add ∫∂ᵦλ between Tᵢ and Tᵢ₊₁ 
        ∇Λ[2]+= 1/model.b *∇λTₖ[2]*(1- δt)- aux/model.b^2 *(1 - δt*( model.b*(jump.time - Tₖ₋₁) +1))
        
        ### Update of ∇λ(Tᵢ₊₁)
        ∇λTₖ = [ δt *(∇λTₖ[1]+1); δt *( -(jump.time - Tₖ₋₁)*aux + ∇λTₖ[2] ); gᵢXₜₖ[i+1,:] ]    
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

