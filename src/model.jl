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


nbdim(model::HawkesStochasticBaseline)::Int = size(model.a,1)

nbparams(model::HawkesStochasticBaseline)::Int = prod(size(model.a))+prod(size(model.b))+prod(size(model.m))
params(model::HawkesStochasticBaseline)::Parameters = [[model.a...;]; [model.b...;]; [model.m...;]]


function params!(model::HawkesStochasticBaseline, θ::Vector) 
    dim = size(model.b)
    model.a =reshape(θ[ 1:ndims(model)^2], (ndims(model),ndims(model)))
    model.b = reshape(θ[ndims(hsb)^1+1: prod(size(hsb.b))+ndims(hsb)^1+1], size(hsb.b))
    model.m = θ[ prod(size(hsb.b))+ndims(hsb)^1+2:end]

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