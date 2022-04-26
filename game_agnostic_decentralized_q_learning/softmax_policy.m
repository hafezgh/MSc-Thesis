function [softmaxv,a,aprob] = softmax_policy( q,tau )
	n = length(q);
    a = zeros(n,1);
    exp_q = exp(q./tau);

    exp_q(exp_q==0) = 1/n;
    if sum(isnan(exp_q),'all')>0
        exp_q(~isinf(exp_q)) = 0;
        exp_q(isinf(exp_q)) = 1/n;
    end
    v = exp_q./sum(exp_q);
    softmaxv = v'*q;
    
    a_pick = sample_from_prob( v );
    a(a_pick) = 1;
    aprob  = v(a_pick);

end


