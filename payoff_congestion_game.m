function [r,s_next] = payoff_congestion_game(s,w,a,c)
    N = size(a,1);
    r = zeros(N,1);
    [GC,GR] = groupcounts(a);
    if max(GC) > N/2
        s_next = 2;
    else
        s_next = 1;
    end
    for i=1:N
        r(i) = w(a(i))*GC(GR==a(i));
    end
    if s == 2
        r = r-c;
    end
end