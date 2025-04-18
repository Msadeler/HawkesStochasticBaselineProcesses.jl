function mle(model::HawkesStochasticBaseline,θ::Union{Nothing,Vector}=nothing, data::DataFrame=DataFrame(); fixed::Union{Vector{Int},Vector{Bool}} = Bool[], method = NelderMead())
    
    if isnothing(θ)
        θ = params(model)
    end
    
    data!(model, data)
    gᵢX!(model, [[[gᵢ(Xₜ) for gᵢ in  model.gₘ.coeff] for Xₜ in df.cov]'...;])
    ∫gᵢX!(model, [ solve(SampledIntegralProblem(model.gᵢX[:,n], model.timedata.time; dim = 1), SimpsonsRule()).u for n in 1:length(model.m)] )

    if fixed isa Vector{Bool}
        res = Int[]
        for (i,v) in enumerate(fixed)
            if v
                push!(res, i)
            end
        end
        fixed = res
    end
    
    unfixed = setdiff(1:length(θ),fixed)
    p = θ[unfixed]


    res = nothing
    print(p)

    function f(θ′)
        θ[unfixed] = θ′
        -likelihood(model, θ,data, model.gᵢX, model.∫gᵢX)
    end

    if method isa Optim.ZerothOrderOptimizer

        res = optimize(f, p, method=method)
    
    elseif method isa Optim.FirstOrderOptimizer
        function g!(storage, θ′)
            θ[unfixed] = θ′
            dlnL = gradient(model, θ)
            storage .= -dlnL[unfixed]
        end
        res = optimize(f, g!, p, method=method)
    elseif method isa Optim.SecondOrderOptimizer
        function g!(storage, θ′)
            θ[unfixed] = θ′
            dlnL = gradient(model, θ)
            storage .= -dlnL[unfixed]
        end
        function h!(storage, θ′)
            θ[unfixed] = θ′
            storage .= -hessian(model, θ, profile=profile)[unfixed, unfixed]
        end
        res = optimize(f, g!, h!, p, method=method)
    end
    p = θ
    p[unfixed] = Optim.minimizer(res)
    params!(model, p)
    return res
end


