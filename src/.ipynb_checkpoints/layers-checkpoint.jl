using Knet
using CUDA

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

mutable struct ConvTranspose2d
    w; b; padding; stride; mode;

    function ConvTranspose2d(w1::Int,w2::Int,cx::Int,cy::Int;padding=0, stride=1, mode=1)
        # w = param(w1,w2,cy,cx; atype=CuArray, init=gaussian(;mean=0.0, std = 0.02))
        w = Param(CuArray(gaussian(w1,w2,cy,cx; mean=0.0, std = 0.02)))
        b = param0(1,1,cy,1; atype=CuArray)
        return new(w,b,padding,stride,mode)
    end
end

function (c::ConvTranspose2d)(x)
    deconv4(c.w, x; padding=c.padding, stride=c.stride, mode=c.mode) .+ c.b
end

mutable struct Conv2d
    w; b; padding; stride; mode;

    function Conv2d(w1::Int,w2::Int,cx::Int,cy::Int;padding=0, stride=1, mode=1)
        # w = param(w1,w2,cx,cy; atype=CuArray)
        w = Param(CuArray(gaussian(w1,w2,cx,cy; mean=0.0, std = 0.02)))
        b = param0(1,1,cy,1; atype=CuArray)
        return new(w,b,padding,stride,mode)
    end
end

function (c::Conv2d)(x)
    conv4(c.w, x; padding=c.padding, stride=c.stride, mode=c.mode) .+ c.b
end

mutable struct Relu
    leaky

    function Relu(;leaky=0)
        return new(leaky)
    end
end

function (rl::Relu)(x)
    if rl.leaky == 0
        return relu.(x)
    else
        return max.(rl.leaky .* x, x)
    end
end

mutable struct Tanh
    function Tanh()
        return new()
    end
end

function (th::Tanh)(x)
    return tanh.(x)
end

mutable struct Sigmoid
    function Sigmoid()
        return new()
    end
end

function (sm::Sigmoid)(x)
    return sigm.(x)
end

mutable struct BatchNorm2d
    bnmoments; bnparams; train_mode; eps;

    function BatchNorm2d(num_features; momentum=0.1, eps=1e-5)
        mom = bnmoments(momentum=momentum)
        par = Param(bnparams(Float32, num_features))
        return new(mom, par, true, eps)
    end
end

function (bn::BatchNorm2d)(x)
    if bn.train_mode
        return batchnorm(x, bn.bnmoments, bn.bnparams, eps=bn.eps, training=bn.train_mode)
    else
        return batchnorm(x, bn.bnmoments, value(bn.bnparams), eps=bn.eps, training=bn.train_mode)
    end
end