import Plots.plot
import Plots.plot!


abstract type TrajectoryPlot end
abstract type IntensityPlot end
abstract type IntensityProcessPlot end



const symbol2typeplot = Dict(
    :T => TrajectoryPlot,
    :I => IntensityPlot,
    :IP => IntensityProcessPlot
)

function plot(hsb::HawkesStochasticBaseline, timedata::DataFrames.DataFrame,type::Symbol=:T)
    plot(hsb, timedata, symbol2typeplot[type])
    
end

function plot(hsb::HawkesStochasticBaseline, timedata::DataFrames.DataFrame, ::Type{TrajectoryPlot})
    bin_cov =0.1
    maxCov, minCov = maximum(timedata.cov)+bin_cov, minimum(timedata.cov)-bin_cov
    db = range(minCov, step = bin_cov, length=trunc(Int, (maxCov-minCov)/bin_cov)+1)
    baselineHeatMap =  reshape( repeat(hsb.baseline.(db), size(timedata.time,1)),:, size(timedata.time,1))
    
    plt = plot(heatmap( timedata.time,db , baselineHeatMap,xlabel="time",color=:deep) )
    plot!(plt, timedata.time, timedata.cov, label="covariable")
    plot!(plt, timedata[timedata.timestamps,:time], timedata[timedata.timestamps,:cov],label="event",  seriestype=:scatter)
    plot!( label=["covariable", "event"])
end

function plot(hsb::HawkesStochasticBaseline, timedata::DataFrames.DataFrame, ::Type{IntensityPlot})

    timedata[!, :baselineValue] = hsb.baseline.(timedata.cov)
    timedata[!, :intensityVal] = timedata[:, :baselineValue]

    intensityJumpTime= timedata[timedata.timestamps,:baselineValue][1]

    lastJump, lastBas = timedata[timedata.timestamps,[:time, :baselineValue]][1,:]

    for simul in eachrow(timedata[(timedata.timestamps).& (timedata.time.>lastJump),:])    

        timedata[(timedata.time .> lastJump).&(timedata.time.<= simul.time),:] = transform!(timedata[(timedata.time .> lastJump).&(timedata.time.<= simul.time),:], [:intensityVal, :time]=> ((i, t) -> i .+ exp.(-hsb.b*( t.-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas))=> :intensityVal)

        intensityJumpTime =  simul.baselineValue + exp(-hsb.b*( simul.time-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas)

        lastJump, lastBas = simul.time, simul.baselineValue

    end
    plot( timedata.time, timedata.intensityVal, label ="intensity")
end


function plot(hsb::HawkesStochasticBaseline, timedata::DataFrames.DataFrame,::Type{IntensityProcessPlot})

    timedata[!, :baselineValue] = hsb.baseline.(timedata.cov)
    timedata[!, :intensityVal] = timedata[:, :baselineValue]

    intensityJumpTime= timedata[timedata.timestamps,:baselineValue][1]

    lastJump, lastBas = timedata[timedata.timestamps,[:time, :baselineValue]][1,:]

    for simul in eachrow(timedata[(timedata.timestamps).& (timedata.time.>lastJump),:])    

        transform!( view( timedata,(timedata.time .> lastJump).&(timedata.time.<= simul.time), : ), [:intensityVal, :time]=> ((i, t) -> i .+ exp.(-hsb.b*( t.-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas))=> :intensityVal)
        intensityJumpTime =  simul.baselineValue + exp(-hsb.b*( simul.time-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas)
        lastJump, lastBas = simul.time, simul.baselineValue
    end
    
    plt = plot( timedata.time, timedata.intensityVal, label ="intensity")
    plot!(plt, timedata[timedata.timestamps .|| timedata.time .== 0, :time ] , 0:size(timedata[timedata.timestamps, :time],1),  linetype=:steppre, label="N")
    
end
