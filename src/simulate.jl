import Base.rand

abstract type MultiDimCov end
abstract type UniDimCov end



const symboltypesimu = Dict(
    :MDC=> MultiDimCov,
    :UDC => UniDimCov
)


function rand(hsb::HawkesStochasticBaseline, condition::Union{Int, Float64})
    rand(hsb, condition, length(hsb.X₀)>1 ? :MDC : :UDC)
end

function rand(hsb::HawkesStochasticBaseline, condition::Union{Int, Float64}, type::Symbol)
    rand(hsb, condition, symboltypesimu[type])
end



function rand(hsb::HawkesStochasticBaseline,maxTime::Float64, ::Type{UniDimCov})::DataFrame

    
    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t₀), :timestamps=> Bool(false), :cov =>hsb.X₀)
    last_timestamp = 0
    cov_val = hsb.X₀
    t=hsb.t₀

    aux = 0
    flag = t < maxTime
    
    while flag
        
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
        

        cov_val = cov_val .+ hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)))
        gₘXₜ = hsb.gₘ(cov_val, hsb.m)

        candidate_intensity = gₘXₜ + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        flag = t_after < maxTime
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition & flag 
            aux = candidate_intensity + hsb.a- gₘXₜ
            flag = t_after < maxTime
            last_timestamp = t_after

        end
        t = t_after
        if flag 

            result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => cov_val)]
        end

    end 


    result= [result;DataFrame(:time => Float64(maxTime), :timestamps=> Bool(false), :cov => hsb.X₀)]
    data!(hsb, result)

    return result

end


function rand(hsb::HawkesStochasticBaseline,nJump::Int,::Type{UniDimCov})::DataFrame

    
    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t₀), :timestamps=> Bool(false), :cov =>hsb.X₀)
    last_timestamp = 0
    cov_val = hsb.X₀
    t=hsb.t₀

    aux = 0
    nSimul = 0
    flag = nSimul   < nJump
    
    while flag
        
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
    
        cov_val = cov_val .+ hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)))

        gₘXₜ = hsb.gₘ(cov_val, hsb.m)
        candidate_intensity = gₘXₜ + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition  
            aux = candidate_intensity + hsb.a- gₘXₜ
            nSimul = nSimul + 1 
            flag = nSimul < nJump
            last_timestamp = t_after

        end
        t = t_after

        result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => cov_val)]
    end 

    data!(hsb, result)
    return result

end

function rand(hsb::HawkesStochasticBaseline,maxTime::Float64, ::Type{MultiDimCov})

    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t₀), :timestamps=> Bool(false), :cov =>Vector[hsb.X₀])
    last_timestamp = 0
    cov_val = hsb.X₀
    t=hsb.t₀

    aux = 0
    flag = t < maxTime
    
    while flag
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
        

        cov_val = cov_val .+ hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)), length(hsb.m))

        gₘXₜ = hsb.gₘ(cov_val, hsb.m)
        candidate_intensity = gₘXₜ + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        flag = t_after < maxTime
        
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition & flag 
            aux = candidate_intensity + hsb.a- gₘXₜ
            flag = t_after < maxTime
            last_timestamp = t_after

        end

        t = t_after
        
        if flag 

            result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => Vector[cov_val])]
        end

    end 


    result= [result;DataFrame(:time => Float64(maxTime), :timestamps=> Bool(false), :cov => Vector[hsb.X₀])]
    data!(hsb, result)

    return result

end


function rand(hsb::HawkesStochasticBaseline,nJump::Int,::Type{MultiDimCov})::DataFrame

    
    #result = (timestamps=Float64[hsb.t0],timeSimu=Float64[hsb.t0],covValue= [Float64[hsb.InitCov]], covTimestamps=[Float64[hsb.InitCov]]  )
    result = DataFrame(:time => Float64(hsb.t₀), :timestamps=> Bool(false), :cov =>Vector[hsb.X₀])
    last_timestamp = 0
    cov_val = hsb.X₀
    t=hsb.t₀

    aux = 0
    nSimul = 0
    flag = nSimul   < nJump
    
    while flag
        
        upper_intensity = hsb.Mmax + aux*(aux>=0)
        t_after = t + rand(Exponential(1 / upper_intensity))
    
        cov_val = cov_val .+ hsb.diffusion(cov_val)  .* (t_after-t) .+ hsb.drift(cov_val) .* rand(Normal(0, sqrt(t_after-t)),length(hsb.m))

        gₘXₜ = hsb.gₘ(cov_val, hsb.m)
        candidate_intensity = gₘXₜ + aux*exp(-hsb.b*(t_after - last_timestamp))
        
        condition = upper_intensity*rand()<= candidate_intensity
        
        if condition  
            aux = candidate_intensity + hsb.a- gₘXₜ
            nSimul = nSimul + 1 
            flag = nSimul < nJump
            last_timestamp = t_after

        end
        t = t_after

        result= [result;DataFrame(:time => t, :timestamps=> condition, :cov => Vector[cov_val])]
    end 

    data!(hsb, result)
    return result

end