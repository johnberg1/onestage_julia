function normalize(x;mean=0.5,std=0.5)
    (x .- mean) ./ std
end

init_optimizer(w,lr,beta1,beta2) = map(x->Adam(;lr=lr,beta1=beta1,beta2=beta2), w)