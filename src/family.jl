abstract type FamilyModel end

using Polynomials



struct UnivariatePolynomialFamilyBaseline <: FamilyModel
    coeff::Function
end

function hessian(x::Float64,μ::Float64;baseline::UnivariatePolynomialFamilyBaseline)
    derivative(derivative(Polynomial(baseline.coeff(x), :μ)))(μ)
end


function gradient(x::Float64,μ::Float64;baseline::UnivariatePolynomialFamilyBaseline)
    derivative(Polynomial(baseline.coeff(x), :μ))(μ)
end

function hessian(x::Float64,μ::Float64;baseline::UnivariatePolynomialFamilyBaseline)
    derivative(derivative(Polynomial(baseline.coeff(x), :μ)))(μ)
end
