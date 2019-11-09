
include("./layer/AbstractLayer.jl")

type ConvolutionLayer{T} <: AbstractLayer
    #
end

function forward{T}(lyr::AffineLayer{T}, x::AbstractArray{T})
    #
end

function backward{T}(lyr::AffineLayer{T}, dout::AbstractArray{T})
    #
end
