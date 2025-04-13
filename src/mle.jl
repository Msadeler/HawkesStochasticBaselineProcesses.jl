function mle(model::HawkesStochasticBaseline, θ::Vector{Float64},  data::DataFrame=DataFrame(); fixed::Union{Vector{Int},Vector{Bool}} = Bool[], method = NelderMead())
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

    function f(θ′)
        θ[unfixed] = θ′
        -likelihood(model, θ,data)
    end

    res = nothing
    print(p)
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