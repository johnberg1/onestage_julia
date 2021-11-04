function normalize(x;mean=0.5,std=0.5)
    (x .- mean) ./ std
end