function [r,s_next] = payoff_distancing(s,w,a,c)
    N = size(a,1);
    r = zeros(N,1);
    [GC,GR] = groupcounts(a);
    if s == 1
        if max(GC) > N/2
            s_next = 2;
        else
            s_next = 1;
        end
    end
    if s == 2
        if max(GC) > N/4
            s_next = 2;
        else
            s_next = 1;
        end
    end

    for i=1:N
        r(i) = w(a(i))*GC(GR==a(i));
    end
    if s == 2
        r = r-c;
    end
end


