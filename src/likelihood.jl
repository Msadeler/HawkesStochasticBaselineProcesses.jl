
function likelihood(model::HawkesStochasticBaseline, θ::Vector{Float64}, df::DataFrame)

    a,b,μ = θ

    transform!(df, [:cov]=> ((cov)-> model.baseline.(cov,μ))=> :baselineValue)
    
    
    lastJump, lastBas = df[df.timestamps,[:time, :baselineValue]][1,:]


    if sum(df.baselineValue.<0)>0 || b <= 0 || a <=0
        -1e30    
    else
        lambdaTk = lastBas
        logIntensity  = log(lambdaTk)

        problem = SampledIntegralProblem(df.baselineValue,df.time; dim = 1)
        method = SimpsonsRule()
        val =solve(problem, method)
        compensator = val.u     ### integrate of baseline along the trajectory of the covariate


        for realisation in eachrow(df[(df.timestamps) .& (df.time.>lastJump),:])

            compensator += (1- exp(-b*(realisation.time - lastJump)))*(lambdaTk + a - lastBas)/b ### compute the compensator
            lambdaTk = realisation.baselineValue + exp(-b*(realisation.time- lastJump))*(lambdaTk+ a - lastBas) ## compute lambda(Tk+1)
            logIntensity+= log(lambdaTk) ## add to the log-intensity

            lastJump, lastBas = realisation.time, realisation.baselineValue 
            
            
        end

        lastSimul = df[end,:]

        compensator += (1- exp(-b*(lastSimul.time - lastJump)))*(lambdaTk + a - lastBas)/b

        return(logIntensity-compensator)        
    end

end


