mutable struct HawkesStochasticBaseline
    a::Float64
    b::Float64
    μ::Float64
    Mmax::Float64   
    drift::Function
    diffusion::Function
    baseline::Function
    t0::Float64
    InitCov::Vector{Float64}
#    MaxTime::Float64
#    MaxJump::Int
end

HawkesStochasticBaseline(a::Real, b::Real, μ::Real=1.0;Mmax::Real=1e15, drift::Function = x->0.0, diffusion::Function=x->0.0, baseline::Function=(x,μ)-> μ, t0::Real=0.0, InitCov::Vector{Float64}=Float64[0.0]) = HawkesStochasticBaseline(a,b, μ , Mmax, drift, diffusion, baseline, t0, InitCov)   


nbparams(m::HawkesStochasticBaseline)::Int = 3

params(m::HawkesStochasticBaseline)::Parameters = [m.a, m.b, m.μ]

function params!(m::HawkesStochasticBaseline, θ::Vector{Float64}) 
    m.a, m.b, m.μ = θ
end 
	