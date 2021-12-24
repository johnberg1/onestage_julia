using Knet
using CUDA

include("layers.jl")

mutable struct DCGenerator
    zdim; d; num_channels; layers;

    function DCGenerator(;zdim=100, d=128, num_channels=1)
        layers = Chain(ConvTranspose2d(4,4,zdim,d*8;stride=1,padding=0),
                        BatchNorm2d(d*8),
                        Relu(),
                        ConvTranspose2d(4,4,d*8,d*4;stride=2,padding=1),
                        BatchNorm2d(d*4),
                        Relu(),
                        ConvTranspose2d(4,4,d*4,d*2;stride=2,padding=1),
                        BatchNorm2d(d*2),
                        Relu(),
                        ConvTranspose2d(4,4,d*2,d;stride=2,padding=1),
                        BatchNorm2d(d),
                        Relu(),
                        ConvTranspose2d(4,4,d,num_channels;stride=2,padding=1),
                        Tanh())
        return new(zdim,d,num_channels,layers)
    end
end

function (g::DCGenerator)(x)
    return g.layers(x)
end

mutable struct DCDiscriminator
    d; num_channels; layers;
    
    function DCDiscriminator(;d=128, num_channels=1)
        layers = Chain(Conv2d(4,4,num_channels,d;stride=2,padding=1),
                        Relu(;leaky=0.2),
                        Conv2d(4,4,d,d*2;stride=2,padding=1),
                        BatchNorm2d(d*2),
                        Relu(;leaky=0.2),
                        Conv2d(4,4,d*2,d*4;stride=2,padding=1),
                        BatchNorm2d(d*4),
                        Relu(;leaky=0.2),
                        Conv2d(4,4,d*4,d*8;stride=2,padding=1),
                        BatchNorm2d(d*8),
                        Relu(;leaky=0.2),
                        Conv2d(4,4,d*8,1;stride=1,padding=0))
                        # Sigmoid())
        return new(d,num_channels,layers)
    end
end

function (d::DCDiscriminator)(x)
    return d.layers(x)
end