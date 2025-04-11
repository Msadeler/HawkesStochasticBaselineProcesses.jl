
function likelihood(theta::Vector,model::Model)

    m = theta[1:end-2]
    a,b = theta[end-2:end]

    transform!(model.timedata, [:cov]=> ((cov)-> model.baseline.(cov,m))=> :baselineValue)

    lastJump, lastBas = model.timedata[model.timedata.timestamps,[:time, :baselineValue]][1,:]


    if sum(model.timedata.baselineValue.<0)>0
        1e15    
    elseif b <= 0 || a <=0
        loglik = 1e15
    else
        lambdaTk = lastBas
        logIntensity  = log(lambdaTk)

        problem = SampledIntegralProblem(model.timedata.baselineValue, model.timedata.time; dim = 1)
        method = SimpsonsRule()
        val =solve(problem, method)
        compensator = val.u         ### integrate of baseline along the trajectory of the covariate


        for realisation in eachrow(model.timedata[(model.timedata.timestamps) .& (model.timedata.time.>lastJump),:])

            print("lastJump ", lastJump, " time ", realisation.time, " baseline ", realisation.baselineValue, " lastbase ", lastBas )

            compensator += (1- exp(-b*(realisation.time - lastJump)))*(lambdaTk + a - lastBas)/b ### compute the compensator
            lambdaTk = realisation.baselineValue + exp(-b*(realisation.time- lastJump))*(lambdaTk+ a - lastBas) ## compute lambda(Tk+1)
            logIntensity+= log(lambdaTk) ## add to the log-intensity
            lastJump, lastBas = realisation.time, realisation.baselineValue 
            
            
        end

        lastSimul = model.timedata[end,:]
        compensator += (1- exp(-b*(lastSimul.time - lastJump)))*(lambdaTk + a - lastBas)/b

        
    end

    return(logIntensity-compensator)
end


