abstract type AbstractFamilyBaseline  end

using Polynomials



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
    coeff::Vector{Function}
end

(gₘ::LinearFamilyBaseline)(x::Union{Real,Vector},m::Vector) = [fi(x) for fi in gₘ.coeff]'*m

function gradient(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, μ::Vector)
    [fi(x) for fi in gₘ.coeff]   
end

function hessian(gₘ::LinearFamilyBaseline, x::Union{Real,Vector}, m::Vector)
    zeros(length(m), length(m))   
end
