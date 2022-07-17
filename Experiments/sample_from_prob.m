function [ i ] = sample_from_prob( prob )
    n = length(prob);
    r = rand;
    c = 0;
    for j = 1:n
        c = c + prob(j);
        if r < c
            i = j;
            return
        end
    end
    i = n;
end

