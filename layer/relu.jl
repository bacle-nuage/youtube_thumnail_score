
include("./layer/AbstractLayer.jl")

type ReluLayer <: AbstractLayer
    mask::AbstractArray{Bool}
    ReluLayer() = new()
end

function forward{T}(lyr::ReluLayer, x::AbstractArray{T})
    mask = lyr.mask = (x .<= 0)
    out = copy(x)
    out[mask] = zero(T)
    out
end

function backward{T}(lyr::ReluLayer, dout::AbstractArray{T})
    dout[lyr.mask] = zero(T)
    dout
end
