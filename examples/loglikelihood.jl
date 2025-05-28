using HawkesStochasticBaselineProcesses 
using LinearAlgebra
using Integrals
using DataFrames


####################################################
#################### 1D Plot #######################
####################################################



x¹ = [0.1,0.1]
gₘ = Baseline( [LinearFamilyBaseline([ x-> 1-exp(-norm(x-x¹)*10), x-> exp(-norm(x-x¹)*10 )]) , LinearFamilyBaseline([ x-> norm(x)/200 ])])

### dXₜ = -b(a-Xₜ)dt + σdWₜ 

function diffusion(x,t)
    z  = t/100%1
    return(30*(z-1/3)*(z<=1/3) + 10*(3*z-2)*(z>=2/3))
end

function drift(x,t)
    z =t/100%1
    return(0.05+10*(z-0.5)^2)
end




function logtest(hsb::HawkesStochasticBaseline, θ::Vector, df::DataFrame )
    
    params!(hsb,θ)
    n = size(hsb).mark


    ### Database with the jump times
    Jumpdb = df[df.timestamps.>=1,:] 


    ### init variables
    Tₖ₋₁ = Jumpdb.time[1]

    Yᵢⱼ = zeros((n,n))
    Yᵢⱼ[:,Jumpdb.timestamps[1]] = hsb.a[:,Jumpdb.timestamps[1]]

    ### Init loglik 
    gₘXₜ  = reduce(hcat,[hsb.gₘ(x,hsb.m) for x in df.cov])
    ∫gₘXₜ = [ solve(SampledIntegralProblem(gₘXₜ[n,:], df.time; dim = 1), SimpsonsRule()).u for n in 1:length(hsb.m)]


    l = log(hsb.gₘ(Jumpdb.cov[1], hsb.m)[Jumpdb.timestamps[1]]) - sum(∫gₘXₜ)

    for jump in eachrow(Jumpdb[2:end,:])

        λTₖ = hsb.gₘ(jump.cov, hsb.m) .+ sum.( eachrow(Yᵢⱼ.*exp.(-hsb.b.*(jump.time - Tₖ₋₁))) )
        Λ =  sum.(eachrow(Yᵢⱼ ./ hsb.b .* (1 .- exp.(-hsb.b.*(jump.time - Tₖ₋₁)))))

        println(λTₖ[jump.timestamps])
        println(Λ)
        println(Yᵢⱼ)

        Yᵢⱼ = Yᵢⱼ.*exp.(-hsb.b.*(jump.time - Tₖ₋₁))
        Yᵢⱼ[:,jump.timestamps] += hsb.a[:,jump.timestamps]
        
        l += log(λTₖ[jump.timestamps]) - sum(Λ)
        Tₖ₋₁ = jump.time
    end

    jump = df[end,:]
    Λ =  sum.(eachrow(Yᵢⱼ ./ hsb.b .* (1 .- exp.(-hsb.b.*(jump.time - Tₖ₋₁)))))
    l -= sum(Λ)


    ####

end



a =0.6*ones(2,2)
b =2 .*[1.0; 1.0]
m = [[1.0],[1.0]]


gₘ = Baseline( [LinearFamilyBaseline([ x-> 1]) , LinearFamilyBaseline([ x-> 1])])

hsb = HawkesStochasticBaseline(a,b,m ;Mmax= 200.0, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )


p = 2000
df = DataFrame( :time => 0:(p+1), :timestamps => [0; fill(1, p); 0], :cov =>fill(1,p+2))



l= loglikelihood(hsb, params(hsb), df )

