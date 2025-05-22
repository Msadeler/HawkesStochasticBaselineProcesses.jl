
function compensator(model::HawkesStochasticBaseline)

    df = copy(model.timedata)

    Λ = zeros(size(df[df.timestamps,:],1) +1 )

    df[:,:gₘXₜ] = [model.gₘ(x, model.m)  for x in df.cov]


    timeJump = df[df.timestamps, :]
    λTₖ = timeJump.gₘXₜ[1]


    Λ[1] = solve(SampledIntegralProblem( df[df.time .<=timeJump.time[1],:gₘXₜ], df[df.time.<=timeJump.time[1],:time]; dim = 1), TrapezoidalRule()).u 

    Tₖ₋₁, gₘXₜₖ₋₁ =  timeJump[1,[:time, :gₘXₜ]]

    for (k,jump) in enumerate(eachrow(timeJump[2:end,:]))

        δt =jump.time- Tₖ₋₁
        Λ[k+1] = (1-exp(-model.b*δt ))*(λTₖ + model.a - gₘXₜₖ₋₁) + solve(SampledIntegralProblem( df[(df.time .<=jump.time) .& (df.time .>= Tₖ₋₁) ,:gₘXₜ], df[(df.time .<=jump.time) .& (df.time .>= Tₖ₋₁) ,:time]; dim = 1), TrapezoidalRule()).u 
        λTₖ = jump.gₘXₜ + exp(-model.b*δt)*(λTₖ + model.a - gₘXₜₖ₋₁)

        Tₖ₋₁ =  jump.time
        gₘXₜₖ₋₁ = jump.gₘXₜ
    end 

    jump = df[end,:]

    if !jump.timestamps
        δt = jump.time - Tₖ₋₁
        Λ[end] = (1-exp(-model.b*δt ))*(λTₖ + model.a - gₘXₜₖ₋₁) + solve(SampledIntegralProblem( df[(df.time .<=jump.time) .& (df.time .>= Tₖ₋₁) ,:gₘXₜ], df[(df.time .<=jump.time) .& (df.time .>= Tₖ₋₁) ,:time]; dim = 1), TrapezoidalRule()).u   
    end


    return(cumsum(Λ))    
end

function compensator(model::HawkesStochasticBaseline,θ::Vector)
    params!(model, θ)
    compensator(model)
end

function compensator(model::HawkesStochasticBaseline,df::DataFrame)
    data!(model, df)
    gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
    ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )
    compensator(model)
end

function compensator(model::HawkesStochasticBaseline,θ::Vector, df::DataFrame)
    data!(model, df)
    gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
    ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )
    params!(model, θ)
    compensator(model)
end
