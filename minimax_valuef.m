function [v] = minimax_valuef(q_vals)
    [m,n] = size(q_vals);
    r = [];
    s = [];
    if min(max(q_vals))==max(min(q_vals'))
        b=max(q_vals);
        for i=1:n
            for j=1:m
                if isequal(b(i),q_vals(j,i))
                    if isequal(q_vals(j,i),min(q_vals(j,:)))
                        r(length(r)+1)=j;
                        s(length(s)+1)=i;
                    end
                end
            end
        end
        v = q_vals(r(1),s(1));
    else
        x = linprog(-[1;zeros(m,1)],[ones(n,1) -q_vals'],zeros(n,1),[0 ones(1,m)],[1],[-inf;zeros(m,1)],...
            [],optimoptions('linprog','Display','none'));
        v = x(1,1);
    end
end