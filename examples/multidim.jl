using HawkesStochasticBaselineProcesses
using Distributions
using LinearAlgebra
using DataFrames


abstract type AbstractFamilyBaseline  end


struct LinearFamilyBaseline <:AbstractFamilyBaseline
    coeff::Vector{Function}
end



f(x)=1

LinearFamilyBaseline([f])

g = reshape([f,f,f,f], (2,2))
LinearFamilyBaseline(g)


g =[[f,f]; [f,f]]
LinearFamilyBaseline(g)

g =[[f,f], [f,f]]
LinearFamilyBaseline(g)


##############################################
##############################################
##############################################
##############################################
##############################################
##############################################

function testsimu(hsb::HawkesStochasticBaseline, max_time::Float64)
    Mmax = 2
    g(x) = [1;1]
    
    
    n = nbdim(hsb)
    
    result = (time=Float64[hsb.t₀],timestamps=Int64[0],cov=[hsb.X₀])    
    
    ## initialisation processus majorant
    a̅ = a.*(a.>=0)
    Y̅ᵢⱼ = zeros(n, n)
    λ𝐌 = Mmax
    
    ### initialisation intensity et temps initial
    
    Yᵢⱼ = zeros(n, n)
    
    Xₖ₋₁ = hsb.X₀
    
    
    t = hsb.t₀
    t⁻ = t

    while t<max_time
    
        t = t⁻ + rand(Exponential(1/sum(λ𝐌)))
    
        Xₜ = Xₖ₋₁ .+ hsb.diffusion(Xₖ₋₁,t).*(t-t⁻) .+ hsb.drift(Xₖ₋₁,t).*rand(Normal(0, sqrt(t-t⁻)))
        #gₘXₜ = hsb.gₘ(Xₜ, hsb.m)
        gₘXₜ = g(Xₜ)
        Yᵢⱼ =  Yᵢⱼ.*exp.(-hsb.b.*(t - t⁻))
    
        
        λ𝐓 =  gₘXₜ .+ sum.(eachrow(Yᵢⱼ))
        
        p =max.(λ𝐓,0)/sum(λ𝐌)
        type_event = argmax(rand(Multinomial(1,[1-sum(p); p])))-1
    
    
        if type_event> 0
    
            Y̅ᵢⱼ  =  Y̅ᵢⱼ.*exp.(-hsb.b.*(t - t⁻))
            Y̅ᵢⱼ[:,type_event] += a̅[:,type_event]
            Yᵢⱼ[:,type_event] += hsb.a[:,type_event]
    
            λ𝐌 = Mmax .+ sum.(eachrow(Y̅ᵢⱼ))
        end
    
        push!(result.time, t)
        push!(result.timestamps, type_event)
        push!(result.cov, Xₜ) 
    
        t⁻ = t
        Xₖ₋₁ = Xₜ
    end 
    

    result = (time= result.time[1:end-1], timestamps = result.timestamps[1:end-1], cov = result.cov[1:end-1] )

    push!(result.cov, result.cov[end] .+ hsb.diffusion(result.cov[end],t)  .* (max_time-result.time[end]) .+ hsb.drift(result.cov[end],t) .* rand(Normal(0, sqrt(max_time-result.time[end]))))
    push!(result.time, max_time)
    push!(result.timestamps, 0)

    df = DataFrame(:time => result.time, :timestamps => result.timestamps, :cov => result.cov)
    data!(hsb, df)
end



a = 0.6*ones(2,2)
b = 2/0.6*a

hsb = HawkesStochasticBaseline(a,b; Mmax=[1.0,1.0])

nb=0

for k in 1:1000

    df =testsimu(hsb,2000.0)
    nb += sum(df.timestamps.>=1)

end



nb/1000

g(x) = [1;1]
    



########## Début fonction


n = nbdim(hsb)
    
result = (time=Float64[hsb.t₀],timestamps=Int64[0],cov=[hsb.X₀])    
    
## initialisation processus majorant
a̅ = a.*(a.>=0)
Y̅ᵢⱼ = zeros(n, n)
λ𝐌 = Mmax

### initialisation intensity et temps initial

Yᵢⱼ = zeros(n, n)

Xₖ₋₁ = hsb.X₀


t = hsb.t₀
t⁻ = t



################################################
################################################
################################################
################################################


sum(λ𝐌)

t = t⁻ + rand(Exponential(1/sum(λ𝐌)))


################## recurrence


Xₜ = Xₖ₋₁ .+ hsb.diffusion(Xₖ₋₁,t).*(t-t⁻) .+ hsb.drift(Xₖ₋₁,t).*rand(Normal(0, sqrt(t-t⁻)))
#gₘXₜ = hsb.gₘ(Xₜ, hsb.m)
gₘXₜ = g(Xₜ)
Yᵢⱼ =  Yᵢⱼ.*exp.(-hsb.b.*(t - t⁻))


λ𝐓 =  gₘXₜ .+ sum.(eachrow(Yᵢⱼ))

p =max.(λ𝐓,0)/sum(λ𝐌)
type_event = argmax(rand(Multinomial(1,[1-sum(p); p])))-1


if type_event> 0

    Y̅ᵢⱼ  =  Y̅ᵢⱼ.*exp.(-hsb.b.*(t - t⁻))
    Y̅ᵢⱼ[:,type_event] += a̅[:,type_event]
    Yᵢⱼ[:,type_event] += hsb.a[:,type_event]

    λ𝐌 = Mmax .+ sum.(eachrow(Y̅ᵢⱼ))
end

push!(result.time, t)
push!(result.timestamps, type_event)
push!(result.cov, Xₜ) 

t⁻ = t
Xₖ₋₁ = Xₜ