@inline function _MinDist(LHC,p)
    n,d = size(LHC)
    mindistp = Inf

    # l-2 norm of distances between all (unique) points
    for i = 2:n
        for j = 1:i-1
            distp = 0.0
            for k = 1:d
                @inbounds dist = LHC[i,k]-LHC[j,k]
                dist -= round(dist/n)*n
                distp += dist^p
            end
            mindistp = min(mindistp, distp)
        end
    end
    mindist = mindistp^(1/p)
    return mindist
end

function _MinDistObjective(dim::Continuous,LHC,p)
    output = _MinDist(LHC,p)
    return output
end

function _MinDistObjective(dim::Categorical,LHC,p)
    output = _MinDist(LHC,p)
    output == Inf ? 0 : output
end

"""
    function MinDistObjective!(LHC::T) where T <: AbstractArray
Return the scalar which should be maximized when using the minimum
distance as the objective function. Note that the minimum distance
is computed as the minimum among all periodical images, e.g. two
points on opposite borders can actually sit close to each other.
"""
function MinDistObjective(LHC::T; p=2,dims::Array{V,1} =[Continuous() for i in 1:size(LHC,2)],
                                            interSampleWeight::Float64=1.0,
                                            ) where T <: AbstractArray where V <: LHCDimension

    out = 0.0

    #Compute the objective function among all points
    out += _MinDistObjective(Continuous(),LHC,p)*interSampleWeight

    #Compute the objective function within each categorical dimension
    categoricalDimInds = findall(x->typeof(x)==Categorical,dims)
    for i in categoricalDimInds
        for j = 1:dims[i].levels
            subLHC = @view LHC[LHC[:,i] .== j,:]
            out += _MinDistObjective(dims[i],subLHC,p)*dims[i].weight
        end
    end

    return out
end

# Remove depwarning in release 2.x.x
function MinDistObjective!(dist,LHC::T; p=2,dims::Array{V,1} =[Continuous() for i in 1:size(LHC,2)],
                                            interSampleWeight::Float64=1.0,
                                            ) where T <: AbstractArray where V <: LHCDimension
    @warn "MinDistObjective!(dist,LHC) is deprecated and does not differ from MinDistObjective(LHC)"
    MinDistObjective(LHC; p = p, dims = dims, interSampleWeight = interSampleWeight)
end
