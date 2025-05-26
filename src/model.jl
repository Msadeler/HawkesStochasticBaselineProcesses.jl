abstract type HawkesProcess  end



mutable struct HawkesStochasticBaseline <: HawkesProcess
    a::Union{Float64,Vector,Matrix}
    b::Union{Float64,Vector,Matrix}
    m::BaselineParameters
    Mmax::Float64   
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

HawkesStochasticBaseline(a::Union{Float64,Vector,Matrix}, b::Union{Float64,Vector,Matrix}, m::Union{Float64,Vector,Matrix}=1.0 ; Mmax::Real=1e15, gₘ::Baseline=Baseline([[LinearFamilyBaseline([x -> 1])]]), drift::Function = (x,t)->0.0, diffusion::Function=(x,t)->0.0, t₀::Real=0.0, X₀::Union{Real,Vector}=0.0,timedata::Nothing=nothing, gᵢX::Union{Nothing,Matrix}=nothing,∫gᵢX::Union{Nothing, Vector}=nothing) = HawkesStochasticBaseline(a,b, m , Mmax,gₘ, drift, diffusion, t₀,X₀,timedata,gᵢX,∫gᵢX)   


nbdim(model::HawkesStochasticBaseline)::Int = size(model.a,1)

nbparams(model::HawkesStochasticBaseline)::Int = prod(size(model.a))+prod(size(model.b))+prod(size(model.m))
params(model::HawkesStochasticBaseline)::Parameters = [[model.a...;]; [model.b...;]; [model.m...;]]


function params!(model::HawkesStochasticBaseline, θ::Vector) 
    model.a =reshape(θ[ 1:ndims(model)^2], (ndims(model),ndims(model)))

    model.b = θ[3:end]
end 

function params!(model::HawkesStochasticBaseline, θ::Vector) 
    model.a, model.b =θ[1:2]
    model.m = θ[3:end]
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