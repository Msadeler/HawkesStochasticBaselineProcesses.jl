mutable struct HawkesStochasticBaseline
    a::Float64
    b::Float64
    Mmax::Float64   
    drift::Function
    diffusion::Function
    baseline::Function
    t0::Float64
    InitCov::Vector{Float64}
#    MaxTime::Float64
#    MaxJump::Int
end

HawkesStochasticBaseline(a::Real, b::Real,Mmax::Real; drift::Function = x->1.0, diffusion::Function=x->1.0, baseline::Function=x-> 1.0, t0::Real=0.0, InitCov::Vector{Float64}=Float64[0.0]) = HawkesStochasticBaseline(a,b,Mmax, drift, diffusion, baseline, t0, InitCov)   
