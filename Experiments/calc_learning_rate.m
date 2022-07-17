function [ a ] = calc_learning_rate(k, C, rho)
    a = C*1/(k)^rho;
end
