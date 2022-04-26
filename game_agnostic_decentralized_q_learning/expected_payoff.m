function [ exp_pays ] = expected_payoff(v,P,state)
    n_states = length(v);
    n_actions = size(P{1,1},1);
    exp_pays = zeros(n_actions,n_actions);
    for next_state = 1:n_states
       exp_pays = exp_pays + v(next_state)*P{next_state,state}; 
    end
end
