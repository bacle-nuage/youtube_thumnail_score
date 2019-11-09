
include("./layer/AbstractLayer.jl")

type AffineLayer{T} <: AbstractLayer
    W::AbstractMatrix{T}
    b::AbstractVector{T}
    x::AbstractArray{T}
    dW::AbstractMatrix{T}
    db::AbstractVector{T}
    function (::Type{AffineLayer}){T}(W::AbstractMatrix{T}, b::AbstractVector{T})
        lyr = new{T}()
        lyr.W = W
        lyr.b = b
        lyr
    end
end

function forward{T}(lyr::AffineLayer{T}, x::AbstractArray{T})
    lyr.x = x
    lyr.W * x .+ lyr.b
end

function backward{T}(lyr::AffineLayer{T}, dout::AbstractArray{T})
    dx = lyr.W' * dout
    lyr.dW = dout * lyr.x'
    lyr.db = _sumvec(dout)
    dx
end
