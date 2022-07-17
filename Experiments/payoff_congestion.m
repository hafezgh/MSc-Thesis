function [r,s_next] = payoff_congestion(s,a,penalty,n_states)
    N = size(a,1);
    r = zeros(N,1);
    [GC,GR] = groupcounts(a);
    a_tmp = a-1;

    if n_states == 257
        encoding = a_tmp(1)*128+a_tmp(2)*64+a_tmp(3)*32+a_tmp(4)*16 ...
            +a_tmp(5)*8+a_tmp(6)*4+a_tmp(7)*2+a_tmp(8)*1;
    end
    if n_states == 17
        encoding = a_tmp(1)*8+a_tmp(2)*4+a_tmp(3)*2+a_tmp(4)*1;
    end

    if s == 1
        for i=1:N
            r(i) = -penalty*GC(GR==a(i));
        end
        s_next = encoding+2;
    end

    if s > 1
        s_next = 1;
    end

    if s ~= 1
        for i=1:N
            for j=1:N
                if j~=i
                    if a(j) == a(i)
                        r(i) = r(i)-penalty;
                    end
                end
            end
        end
    end
end