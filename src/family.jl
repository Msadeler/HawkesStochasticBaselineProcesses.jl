abstract type AbstractFamilyBaseline  end


struct UnivariatePolynomialFamilyBaseline <:AbstractFamilyBaseline
    coeff::Vector{Function}
end

(gₘ::UnivariatePolynomialFamilyBaseline)(x::Real,m::Real) = Polynomial(gₘ.coeff(x), :m)(m)


function hessian(gₘ::UnivariatePolynomialFamilyBaseline,x::Real,μ::Real)
    derivative(derivative(Polynomial(gₘ.coeff(x), :m)))(m)
end


function gradient(gₘ::UnivariatePolynomialFamilyBaseline,x::Float64,μ::Float64)
    derivative(Polynomial(gₘ.coeff(x), :m))(m)
end



mutable struct LinearFamilyBaseline <:AbstractFamilyBaseline
    coeff::Union{Vector{<:Function}, Vector{<:Vector{<:Function}}}
    gX::Union{Nothing,Matrix{Float64}}
    ∫gX::Union{Nothing,Vector{Float64}}
end

LinearFamilyBaseline(coeff::Union{Vector{<:Function}, Vector{<:Vector{<:Function}}};gX::Union{Nothing,Vector{Float64}} =nothing, ∫gX::Union{Nothing,Matrix{Float64}}=nothing) = LinearFamilyBaseline(coeff, gX, ∫gX)


(gₘ::LinearFamilyBaseline)(x::Union{Float64,Vector},m::SubBaselineParameters) = dot([gᵢ(x) for gᵢ in gₘ.coeff],m) 



function prepareintegral!(gₘ::LinearFamilyBaseline, df::DataFrame)
    
    if isnothing(gₘ.gX)
        gₘ.gX = [gᵢ(x) for x in df.cov, gᵢ in gₘ.coeff]
    end

    if isnothing(gₘ.∫gX)
        gₘ.∫gX = [ solve(SampledIntegralProblem(gₘ.gX[:,i], df.time; dim = 1), SimpsonsRule()).u for i in eachindex(gₘ.coeff)]
    end
end

function integral(gₘ::LinearFamilyBaseline,m::SubBaselineParameters)
    gₘ.∫gX ⋅ m
end


function gradient(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, m::SubBaselineParameters)
    [fi(x) for fi in gₘ.coeff]   
end

function hessian(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, m::SubBaselineParameters)
    zeros(length(m), length(m))   
end



struct Baseline 
    coeff::Vector{AbstractFamilyBaseline}
end

function prepareintegral!(gₘ::Baseline, df::DataFrame)
    
    for i in eachindex(gₘ.coeff)
        prepareintegral!(gₘ.coeff[i], df) 
    end

end


(g::Baseline)(x::Union{<:Real,Vector},m::BaselineParameters) = [g.coeff[i](x, m[i])  for i=eachindex(m)]