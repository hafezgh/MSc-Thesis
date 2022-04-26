function [ ee ] = tau_to_zero_decay(k, tau_bar, rho_alpha, rho, D)    
    ee = (tau_bar/(1+(tau_bar*rho_alpha*rho*log(k))/(4*D)));
end

