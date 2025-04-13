import Base.rand




function rand(hsb::HawkesStochasticBaseline,maxTime::Float64)::DataFrame

    
    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t0), :timestamps=> Bool(false), :cov => Vector{Float64}(hsb.InitCov))
    last_timestamp = 0
    cov_val = hsb.InitCov
    t=hsb.t0

    aux = 0
    flag = t < maxTime
    
    while flag
        
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
        

        #push!(result.timeSimu, t_after)
        #push!( result.covValue, hsb.diffusion(result.covValue[end])  .* (t_after-t) .+ hsb.drift(result.covValue[end]) .* rand(Normal(0, sqrt(t_after-t))) )
        cov_val = hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)))
        
        mu_x = hsb.baseline(cov_val, hsb.μ)
        candidate_intensity = mu_x + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        flag = t_after < maxTime
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition & flag 
            aux = candidate_intensity + hsb.a- mu_x
            flag = t_after < maxTime
            last_timestamp = t_after

        end
        t = t_after
        if flag 

            result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => cov_val)]
        end

    end 


    result= [result;DataFrame(:time => Float64(maxTime), :timestamps=> Bool(false), :cov => Float64[cov_val])]

    return result

end


function rand(hsb::HawkesStochasticBaseline,nJump::Int)::DataFrame

    
    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t0), :timestamps=> Bool(false), :cov => Vector{Float64}(hsb.InitCov))
    last_timestamp = 0
    cov_val = hsb.InitCov
    t=hsb.t0

    aux = 0
    nSimul = 0
    flag = nSimul   < nJump
    
    while flag
        
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
        

        #push!(result.timeSimu, t_after)
        #push!( result.covValue, hsb.diffusion(result.covValue[end])  .* (t_after-t) .+ hsb.drift(result.covValue[end]) .* rand(Normal(0, sqrt(t_after-t))) )
        
        cov_val = hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)))
        
        mu_x = hsb.baseline(cov_val, hsb.μ)
        candidate_intensity = mu_x + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition  
            aux = candidate_intensity + hsb.a- mu_x
            nSimul = nSimul + 1 
            flag = nSimul < nJump
            last_timestamp = t_after

        end
        t = t_after

        result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => cov_val)]
    end 


    return result

end