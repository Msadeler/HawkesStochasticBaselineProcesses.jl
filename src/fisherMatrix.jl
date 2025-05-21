
function fisher(model::HawkesStochasticBaseline, data::DataFrame)

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

    gᵢXₜₖ=  model.gᵢX[df.timestamps,:]
    ∇λTₖ = [0.0 ;0.0;gᵢXₜₖ[1,:] ]
    λTₖ = model.gₘ( timeJump.cov[1], model.m)

    Γ = ∇λTₖ*∇λTₖ' / λTₖ^2


    Tₖ₋₁, gₘXₜₖ₋₁ =  timeJump[1,[:time, :gₘXₜ]]


    for (k,jump) in enumerate(eachrow(timeJump[2:end, :]))


        ∇λTₖ[[1,2]] = exp(-model.b*(jump.time -  Tₖ₋₁))* (∇λTₖ[[1,2]] .+  [ 1 ; -(λTₖ + model.a - jump.gₘXₜ)*(jump.time - Tₖ₋₁) ])  

        ∇λTₖ[3:end] = gᵢXₜₖ[k+1,:]

        λTₖ = jump.gₘXₜ +  exp(-model.b*(jump.time -  Tₖ₋₁))*( λTₖ + model.a - gₘXₜₖ₋₁ )

        Γ += ∇λTₖ*∇λTₖ' / λTₖ^2

        Tₖ₋₁ = jump.time
        gₘXₜₖ₋₁ =  jump.gₘXₜ


        
    end

    return(Γ/model.timedata.time[end])
    
    
end
