function [ ee ] = tau_to_eps_decay(k, tau_bar, eps_to)
    ee = (1/k)*tau_bar+(1-1/k)*eps_to;
end

