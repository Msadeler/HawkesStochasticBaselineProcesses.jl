abstract type HawkesProcess  end



mutable struct HawkesStochasticBaseline <: HawkesProcess
    a::Union{Float64,Vector,Matrix}
    b::Union{Float64,Vector,Matrix}
    m::BaselineParameters
    Mmax::Union{Float64,Vector}   
    #gₘ::Union{<:AbstractFamilyBaseline,<:Function}
    gₘ::Baseline
    drift::Function
    diffusion::Function
    t₀::Float64
    X₀::Union{Real,Vector}
    timedata::Union{Nothing, DataFrame}
    gᵢX::Union{Nothing, Matrix}
    ∫gᵢX::Union{Nothing, Vector}
end

HawkesStochasticBaseline(a::Union{Float64,Vector,Matrix}, b::Union{Float64,Vector,Matrix}, m::BaselineParameters=1.0 ; Mmax::Union{<:Real,Vector}   =1e15, gₘ::Baseline=Baseline([[LinearFamilyBaseline([x -> 1])]]), drift::Function = (x,t)->0.0, diffusion::Function=(x,t)->0.0, t₀::Real=0.0, X₀::Union{Real,Vector}=0.0,timedata::Nothing=nothing, gᵢX::Union{Nothing,Matrix}=nothing,∫gᵢX::Union{Nothing, Vector}=nothing) = HawkesStochasticBaseline(a,b, m , Mmax,gₘ, drift, diffusion, t₀,X₀,timedata,gᵢX,∫gᵢX)   


function Base.size(model::HawkesStochasticBaseline)
    (mark=size(model.a,1), dim=length.(model.m)  )
end

Base.length(model::HawkesStochasticBaseline)::Int = prod(size(model.a))+prod(size(model.b))+ sum(length.(model.m))
params(model::HawkesStochasticBaseline)::Parameters = [[model.a...;]; [model.b...;]; [model.m...;]]


function params!(hsb::HawkesStochasticBaseline, θ::Vector) 

    n = size(hsb)

    hsb.a =reshape(θ[ 1:n.mark^2], (n.mark,n.mark) )
    hsb.b = reshape(θ[n.mark^2+1: prod(size(hsb.b))+n.mark^2], size(hsb.b))

    strat = fill(1,length(hsb.m)) + [0, cumsum(length.(hsb.m))[1:end-1]...]
    inter = (:).(strat, strat .+ length.(hsb.m) .-1) 
    hsb.m .= map(x -> θ[prod(size(hsb.b))+n.mark^2 .+ x], inter )


end 


	
function data!(model::HawkesProcess, data::DataFrame)
    model.timedata = data
end

function gᵢX!(model::HawkesProcess, data::Matrix)
    model.gᵢX = data
end

function ∫gᵢX!(model:: HawkesProcess, data::Vector)
    model.∫gᵢX = data
end