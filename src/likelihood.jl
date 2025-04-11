
function Likelihood(theta::Vector,model::model)

    m = theta[1:end-2]
    a,b = theta[end-2:end]

    transform!(timedata, [:cov]=> ((cov)-> baseline.(cov,m))=> :baselineValue)

    lastJump, lastBas = timedata[timedata.timestamps,[:time, :baselineValue]][1,:]


    if sum(timedata.baselineValue.<0)>0
        loglik= 1e15
    end

    if b <= 0 || a <=0
        log = 1e15
    else
        lambdaTk = lastBas
        logIntensity  = log(lambdaTk)

        problem = SampledIntegralProblem(timedata.baselineValue, timedata.time; dim = 1)
        method = SimpsonsRule()
        val =solve(problem, method)
        compensator = val.u         ### integrate of baseline along the trajectory of the covariate


        for realisation in eachrow(timedata[(timedata.timestamps) .& (timedata.time.>lastJump),:])

            print("lastJump ", lastJump, " time ", realisation.time, " baseline ", realisation.baselineValue, " lastbase ", lastBas )

            compensator += (1- exp(-b*(realisation.time - lastJump)))*(lambdaTk + a - lastBas)/b ### compute the compensator
            lambdaTk = realisation.baselineValue + exp(-b*(realisation.time- lastJump))*(lambdaTk+ a - lastBas) ## compute lambda(Tk+1)
            logIntensity+= log(lambdaTk) ## add to the log-intensity
            lastJump, lastBas = realisation.time, realisation.baselineValue 
            
            
        end

        lastSimul = timedata[end,:]
        compensator += (1- exp(-b*(lastSimul.time - lastJump)))*(lambdaTk + a - lastBas)/b

        
    end

    logIntensity-compensator
    
end


