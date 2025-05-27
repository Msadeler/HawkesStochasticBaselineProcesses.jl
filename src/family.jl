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



struct LinearFamilyBaseline <:AbstractFamilyBaseline
    coeff::Union{Matrix{<:Function}, Vector{<:Function}, Vector{<:Vector{<:Function}}}
end


(gₘ::LinearFamilyBaseline)(x::Union{Real,Vector},m::SubBaselineParameters) = dot([gᵢ(x) for gᵢ in gₘ.coeff],m) 

function gradient(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, m::SubBaselineParameters)
    [fi(x) for fi in gₘ.coeff]   
end

function hessian(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, m::SubBaselineParameters)
    zeros(length(m), length(m))   
end



struct Baseline 
    coeff::Vector{Vector{AbstractFamilyBaseline}}
end


(g::Baseline)(x::Union{<:Real,Vector},m::BaselineParameters) = [f(x, m[i])  for i=eachindex(m) for f in g.coeff[i]]