
function likelihood(model::HawkesStochasticBaseline, θ::Vector{Float64}, df::DataFrame, gᵢX::Matrix,∫gᵢX::Vector)

    params!(model,θ)

    df[!,:baselineValue] = gᵢX*model.m
    
    
    lastJump, lastBas = df[df.timestamps,[:time, :baselineValue]][1,:]


    if sum(df.baselineValue.<0)>0 || model.b <= 0 || model.a <=0
        -1e30    
    else
        lambdaTk = lastBas
        logIntensity  = log(lambdaTk)

        compensator =  ∫gᵢX'*model.m   ### integrate of baseline along the trajectory of the covariate


        for realisation in eachrow(df[(df.timestamps) .& (df.time.>lastJump),:])

            compensator += (1- exp(-model.b*(realisation.time - lastJump)))*(lambdaTk + model.a - lastBas)/model.b ### compute the compensator
            lambdaTk = realisation.baselineValue + exp(-model.b*(realisation.time- lastJump))*(lambdaTk+ model.a - lastBas) ## compute lambda(Tk+1)
            logIntensity+= log(lambdaTk) ## add to the log-intensity

            lastJump, lastBas = realisation.time, realisation.baselineValue 
            
            
        end

        lastSimul = df[end,:]

        compensator += (1- exp(-model.b*(lastSimul.time - lastJump)))*(lambdaTk + model.a - lastBas)/model.b

        return(logIntensity-compensator)        
    end

end


function gradient(model::HawkesStochasticBaseline, θ::Vector, df::DataFrame)


end