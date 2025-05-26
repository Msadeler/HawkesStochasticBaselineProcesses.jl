abstract type ExpTest end
abstract type UniTest end

const symbol2typetest = Dict(
    :E => ExpTest,
    :U => UniTest
)

function processAgregation(timeransformedList::Vector)

        aggregatedProcess = timeransformedList[1][1:end-1]
        maxTime = timeransformedList[1][end]

        for list in timeransformedList[2:end]

            aggregatedProcess = [aggregatedProcess; maxTime .+ list[1:end-1]]
            maxTime += list[end]
            
        end

        return(aggregatedProcess, maxTime)
end

function procedureGOF(timeransformedList::Vector, type::Symbol =:U ; options...)
    procedureGOF(timeransformedList, symbol2typetest[type]; options...)
    
end

function procedureGOF(timeTransformedList::Vector, ::Type{UniTest}; supΛ::Union{Nothing,Real}=nothing)

    aggProcess , maxTime = processAgregation(timeTransformedList)

    if isnothing(supΛ)
        supΛ = 0.9*maxTime
    end

    if supΛ> maxTime
        print("The upper bound given is greater than the upper bound of the aggregated process")
    end

    pvalue(ExactOneSampleKSTest(aggProcess[aggProcess.<= supΛ]/supΛ,Uniform(0,1)))
    
    
end

function procedureGOF(timeransformedList::Vector, ::Type{ExpTest})
    
    aggProcess , maxTime = processAgregation(timeransformedList)

    pvalue(ExactOneSampleKSTest( aggProcess[2:end].- aggProcess[1:end-1], Exponential(1)))
    
end








