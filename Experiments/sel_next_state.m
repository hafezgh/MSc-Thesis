function [ next_state ] = sel_next_state( pp,state,a1,a2 )
    [~,n] = size(pp);
    prob = zeros(1,n);
    for i = 1:n
        prob(i) = a1'*pp{i,state}*a2;
    end
    next_state = sample_from_prob(prob);
end

