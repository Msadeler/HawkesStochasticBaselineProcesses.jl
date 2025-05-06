abstract type HawkesProcess  end


mutable struct HawkesStochasticBaseline <: HawkesProcess
    a::Float64
    b::Float64
    m::Union{Vector,Real}
    Mmax::Float64   
    gₘ::Union{<:AbstractFamilyBaseline,<:Function}
    drift::Function
    diffusion::Function
    t₀::Float64
    X₀::Union{Real,Vector}
    timedata::Union{Nothing, DataFrame}
    gᵢX::Union{Nothing, Matrix}
    ∫gᵢX::Union{Nothing, Vector}
#    MaxTime::Float64
#    MaxJump::Int
end

HawkesStochasticBaseline(a::Real, b::Real, m::Union{Vector,Real}=1.0 ; Mmax::Real=1e15, gₘ::Union{<:AbstractFamilyBaseline,<:Function}=(x,m)-> m, drift::Function = (x,t)->0.0, diffusion::Function=(x,t)->0.0, t₀::Real=0.0, X₀::Union{Real,Vector}=[0.0],timedata::Nothing=nothing, gᵢX::Union{Nothing,Matrix}=nothing,∫gᵢX::Union{Nothing, Vector}=nothing) = HawkesStochasticBaseline(a,b, m , Mmax,gₘ, drift, diffusion, t₀,X₀,timedata,gᵢX,∫gᵢX)   


mutable struct MultidimHawkesStochasticBaseline <: HawkesProcess
    a::Matrix
    b::Matrix
    m::Union{Vector,Matrix}
    Mmax::Float64   
    gₘ::Union{<:AbstractFamilyBaseline,<:Function}
    drift::Function
    diffusion::Function
    t₀::Float64
    X₀::Union{Real,Vector}
    timedata::Union{Nothing, DataFrame}
    gᵢX::Union{Nothing, Matrix}
    ∫gᵢX::Union{Nothing, Vector}
end

MultidimHawkesStochasticBaseline(a::Matrix, b::Matrix, m::Union{Vector,Real,Matrix}=1.0 ; Mmax::Real=1e15, gₘ::Union{<:AbstractFamilyBaseline,<:Function}=(x,m)-> m, drift::Function = (x,t)->0.0, diffusion::Function=(x,t)->0.0, t₀::Real=0.0, X₀::Union{Real,Vector}=[0.0],timedata::Nothing=nothing, gᵢX::Union{Nothing,Matrix}=nothing,∫gᵢX::Union{Nothing, Vector}=nothing) = MultidimHawkesStochasticBaseline(a,b, m , Mmax,gₘ, drift, diffusion, t₀,X₀,timedata,gᵢX,∫gᵢX)   


nbdim(model::MultidimHawkesStochasticBaseline)::Int = size(model.a,1)

nbparams(model::HawkesStochasticBaseline)::Int = 2 + length(model.m)
nbparams(model::MultidimHawkesStochasticBaseline)::Int = prod(size(model.a))+prod(size(model.b))+prod(size(model.m))


params(model::HawkesStochasticBaseline)::Parameters = [model.a; model.b; model.m]
params(model::MultidimHawkesStochasticBaseline)::Parameters = [[model.a...;]; [model.b...;]; [model.m...;]]


function params!(model::MultidimHawkesStochasticBaseline, θ::Vector) 
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