

function OneSampleTTest(model::HawkesStochasticBaseline,μ₀::Real,coeff_index::Int64)
    

    T = model.timedata.time[end]
    Σ = diag(inv(fisher(model, model.timedata)))
    Z = sqrt(T)*(params(model)[coeff_index]- μ₀)/sqrt(Σ[coeff_index])

    Dict( "Statistics"=> Z, "pval" => 2*(1- cdf(Normal(0,1), abs(Z))) )

    
end


function EqualCoeffTest(model::HawkesStochasticBaseline,index₁::Int64,index₂::Int64)
    

    T = model.timedata.time[end]
    Σ = inv(fisher(model, model.timedata))
    Z = sqrt(T)*(params(model)[index₁]-params(model)[index₂])/ sqrt( Σ[index₁,index₁] + Σ[index₂,index₂] - 2*Σ[index₁,index₂] )

    Dict( "Statistics"=> Z, "pval" => 2*(1- cdf(Normal(0,1), abs(Z))) )

    
end