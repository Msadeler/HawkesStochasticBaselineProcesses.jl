
abstract type TrajectoryPlot end
abstract type IntensityPlot end
abstract type IntensityProcessPlot end
abstract type TrajectoryPlotProjected end

const symbol2typeplot = Dict(
    :T => TrajectoryPlot,
    :TP => TrajectoryPlotProjected,
    :I => IntensityPlot,
    :IP => IntensityProcessPlot
)



function plot(hsb::HawkesStochasticBaseline,type::Symbol=:T; options...)

    if type == :T
        dim = length(hsb.X₀)>1 ? :MDC : :UDC

        plot(hsb, symbol2typeplot[type], symboltypecov[dim]; options...)
    else
        plot(hsb, type; options...)
    end
end

function plot(hsb::HawkesStochasticBaseline, ::Type{TrajectoryPlot},::Type{UniDimCov}; bin_cov::Float64=0.1, bin_time::Float64=0.1)


    maxCov, minCov = maximum(hsb.timedata.cov)+bin_cov, minimum(hsb.timedata.cov)-bin_cov
    xgrid = minCov-bin_cov:bin_cov: maxCov+bin_cov
    tgrid = hsb.timedata.time[1]:bin_time:hsb.timedata.time[end]
    baselineHeatMap =  transpose(reshape( repeat(hsb.gₘ.(xgrid, hsb.m), length(tgrid)),:, length(tgrid)))
    
    
    f = Figure()
    ax, hm = heatmap(f[1,1][1,1],tgrid, xgrid, baselineHeatMap,colormap=:deep)
    Colorbar(f[1,1][1,2], hm)  
    lines!(hsb.timedata.time, hsb.timedata.cov)
    scatter!(hsb.timedata[hsb.timedata.timestamps,:time], hsb.timedata[hsb.timedata.timestamps,:cov], color=(:orange), markersize=7)
    f
end

function plot(hsb::HawkesStochasticBaseline, ::Type{IntensityPlot})

    hsb.timedata[!, :baselineValue] = hsb.gₘ.(timedata.cov,hsb.m)
    hsb.timedata[!, :intensityVal] = hsb.timedata[:, :baselineValue]

    intensityJumpTime= hsb.timedata[hsb.timedata.timestamps,:baselineValue][1]

    lastJump, lastBas = hsb.timedata[hsb.timedata.timestamps,[:time, :baselineValue]][1,:]

    for simul in eachrow(hsb.timedata[(hsb.timedata.timestamps).& (hsb.timedata.time.>lastJump),:])    

        hsb.timedata[(hsb.timedata.time .> lastJump).&(hsb.timedata.time.<= simul.time),:] = transform!(hsb.timedata[(hsb.timedata.time .> lastJump).&(hsb.timedata.time.<= simul.time),:], [:intensityVal, :time]=> ((i, t) -> i .+ exp.(-hsb.b*( t.-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas))=> :intensityVal)

        intensityJumpTime =  simul.baselineValue + exp(-hsb.b*( simul.time-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas)

        lastJump, lastBas = simul.time, simul.baselineValue

    end
    plot( hsb.timedata.time, hsb.timedata.intensityVal, label ="intensity")
end


function plot(hsb::HawkesStochasticBaseline,::Type{IntensityProcessPlot})

    hsb.timedata[!, :baselineValue] = hsb.gₘ.(hsb.timedata.cov,hsb.m)
    hsb.timedata[!, :intensityVal] = hsb.timedata[:, :baselineValue]

    hsb.intensityJumpTime= hsb.timedata[hsb.timedata.timestamps,:baselineValue][1]

    lastJump, lastBas = hsb.timedata[hsb.timedata.timestamps,[:time, :baselineValue]][1,:]

    for simul in eachrow(hsb.timedata[(hsb.timedata.timestamps).& (hsb.timedata.time.>lastJump),:])    

        transform!( view( hsb.timedata,(hsb.timedata.time .> lastJump).&(hsb.timedata.time.<= simul.time), : ), [:intensityVal, :time]=> ((i, t) -> i .+ exp.(-hsb.b*( t.-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas))=> :intensityVal)
        intensityJumpTime =  simul.baselineValue + exp(-hsb.b*( simul.time-lastJump)).*(intensityJumpTime .+ hsb.a .- lastBas)
        lastJump, lastBas = simul.time, simul.baselineValue
    end
    
    plt = plot( hsb.timedata.time, hsb.timedata.intensityVal, label ="intensity")
    plot!(plt, hsb.timedata[hsb.timedata.timestamps .|| hsb.timedata.time .== 0, :time ] , 0:size(hsb.timedata[hsb.timedata.timestamps, :time],1),  linetype=:steppre, label="N")
    
end


function plot(model::HawkesStochasticBaseline,  ::Type{TrajectoryPlot}, ::Type{MultiDimCov};dim_1::Int64=1, dim_2::Int64=2, bin_cov::Float64 = 0.001)

    Xₜ= [model.timedata.cov'...;]
    Xₜₖ = [model.timedata[model.timedata.timestamps, :cov]'...;]

    xgrid = minimum(Xₜ[:,dim_1])-bin_cov:bin_cov: maximum(Xₜ[:,dim_1])+bin_cov
    ygrid =  minimum(Xₜ[:,dim_2])-bin_cov:bin_cov: maximum(Xₜ[:,dim_2])+bin_cov

    gₘX = [[model.gₘ([x,y], model.m) for x in xgrid] for y in ygrid]
    gₘX = [gₘX'...; ]


    f = Figure()

    ax, hm = heatmap(f[1,1][1,1],xgrid, ygrid,gₘX, colormap =:deep)
    Colorbar(f[1,1][1,2], hm)  
    lines!(Xₜ[:,dim_1], Xₜ[:,dim_2], color=(:black, 0.7),linewidth=0.8)
    #scatter!(Xₜₖ[:,dim_1], Xₜₖ[:,dim_2], color=(:orange, 1),markersize=4)

    f
end

