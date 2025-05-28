import Base.rand


function rand(hsb::HawkesStochasticBaseline, max_time::Float64)

    n = nbdim(hsb)
    
    result = (time=Float64[hsb.t₀],timestamps=Int64[0],cov=[hsb.X₀])    
    
    ## initialisation processus majorant
    a̅ = hsb.a.*(hsb.a.>=0)
    Y̅ᵢⱼ = zeros(n, n)
    λ𝐌 = hsb.Mmax
    
    ### initialisation intensity et temps initial
    
    Yᵢⱼ = zeros(n, n)
    Xₖ₋₁ = hsb.X₀
    
    
    t = hsb.t₀
    t⁻ = t

    while t<max_time
    
        t = t⁻ + rand(Exponential(1/sum(λ𝐌)))
    
        ## update covariable
        Xₜ = Xₖ₋₁ .+ hsb.diffusion(Xₖ₋₁,t).*(t-t⁻) .+ hsb.drift(Xₖ₋₁,t).*rand(Normal(0, sqrt(t-t⁻)))

        ## update baseline value and kernel value
        gₘXₜ = hsb.gₘ(Xₜ,hsb.m)
        Yᵢⱼ =  Yᵢⱼ.*exp.(-hsb.b.*(t - t⁻))
    
        ## intensity value at the simulated time
        λ𝐓 =  gₘXₜ .+ sum.(eachrow(Yᵢⱼ))
        
        ## Simulation of the component jumping. type_event=0 imply no jumps at all
        p =max.(λ𝐓,0)/sum(λ𝐌)
        type_event = argmax(rand(Multinomial(1,[1-sum(p); p])))-1
    
    
        if type_event> 0  ## update of variable if ther is a jump
    
            Y̅ᵢⱼ  =  Y̅ᵢⱼ.*exp.(-hsb.b.*(t - t⁻))
            Y̅ᵢⱼ[:,type_event] += a̅[:,type_event]
            Yᵢⱼ[:,type_event] += hsb.a[:,type_event]

            λ𝐌 = hsb.Mmax .+ sum.(eachrow(Y̅ᵢⱼ))
        end
    
        ## conserve the data
        push!(result.time, t)
        push!(result.timestamps, type_event)
        push!(result.cov, Xₜ) 
    
        t⁻ = t
        Xₖ₋₁ = Xₜ
    end 
    

    ## formatages des données en dataframe
    result = (time= result.time[1:end-1], timestamps = result.timestamps[1:end-1], cov = result.cov[1:end-1] )

    push!(result.cov, result.cov[end] .+ hsb.diffusion(result.cov[end],t)  .* (max_time-result.time[end]) .+ hsb.drift(result.cov[end],t) .* rand(Normal(0, sqrt(max_time-result.time[end]))))
    push!(result.time, max_time)
    push!(result.timestamps, 0)

    df = DataFrame(:time => result.time, :timestamps => result.timestamps, :cov => result.cov)
    data!(hsb, df)
end