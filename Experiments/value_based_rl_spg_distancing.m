clear all; close all; clc;
tic

rng(517);
trial_num = 8;

n_players = 8;
n_states = 2; 
penalty=n_players*10; % penalty for being in the unsafe state
n_actions = 4; w = [0.05,0.1,0.8,1.6]; % number of actions per state (facilities) and their weights

num_iter =1*10^3;
err = 1e-6;

interval = 1;

interval_print = 10^5;
num_save_iters = num_iter/(interval);
iter_save = 1:interval:num_iter;

gamma = 0.5;
rho_alpha = 0.9;
rho_beta = 1.0;
rho = 0.70;
tau_bar_zero = 0.25;
tau0 = 2.5;
DD = 1/(1-gamma);
C=1.5;
%% Iterations

v_iter = zeros(num_save_iters,n_players,n_states,trial_num);
q_iter = zeros(num_save_iters,n_players,n_actions,n_states,trial_num);
policy_iter = zeros(num_save_iters,n_players,n_actions,n_states,trial_num);


parfor trial = 1:trial_num

    % Initialization
    state = 1; % Initial state
    tau = tau0.*ones(n_states,1);
    q = zeros(n_players,n_actions,n_states);
    v = zeros(n_players,n_states,1);
    v_iter_tmp = zeros(num_save_iters,n_players,n_states);
    q_iter_tmp = zeros(num_save_iters,n_players,n_actions,n_states);
    policy_tmp = zeros(num_save_iters,n_players,n_actions,n_states);
    % Iterations
    state_counts=ones(1,n_states);
    
    
    
    for k = 1:num_iter
      
        if mod(k,interval_print) == 0
            k/(interval_print)
            
        end

        tau_old = tau(state);
        tau(state) = tau_to_zero_decay(state_counts(state), tau_bar_zero, rho_alpha, rho, DD);
        actions = zeros(n_players,1);
        softmaxvs = zeros(n_players,1);
        a_probs = zeros(n_players,1);
        alpha_bars = zeros(n_players,1);
        
        for i=1:n_players
            [softmaxv, a, a_prob,br] = softmax_policy( q(i,:,state)',tau_old );
            actions(i)=find(a);
            softmaxvs(i) = softmaxv;
            a_probs(i) = a_prob;
            alpha_bars(i) = min(1,calc_learning_rate(state_counts(state),C,rho_alpha)/a_prob);
        end
        
        [r,next_state] = payoff_distancing(state,w,actions,penalty);
        
        for i=1:n_players
            q(i, actions(i),state) = q(i, actions(i),state)+ alpha_bars(i)*...
            (r(i) + gamma*v(i,next_state) - q(i, actions(i),state)); 
            v(i,state) = v(i,state) +...
                calc_learning_rate(state_counts(state),C,rho_beta)*(softmaxvs(i)-v(i,state));
        end
        
        if sum(find(iter_save==k))>0
            jj = find(iter_save==k);
            v_iter_tmp(jj,:,:) = v;
            q_iter_tmp(jj,:,:,:) = q;
            for pl=1:n_players
                for s=1:n_states
                   expq = exp(q_iter_tmp(jj,pl,:,s)./tau(s));
                   policy_tmp(jj,pl,:,s) = expq./sum(expq);
                end
            end
        
        end
        
        state_counts(state)=state_counts(state)+1;
        state = next_state;

    end
    
    v_iter(:,:,:,trial) = v_iter_tmp;
    q_iter(:,:,:,:,trial) = q_iter_tmp;
    policy_iter(:,:,:,:,trial) = policy_tmp;
    
end

%% Post-processing and Plotting
l1_acc = zeros(num_save_iters,trial_num);
l1_acc_pl = zeros(n_players,num_save_iters,trial_num);

converged = zeros(trial_num,n_players);
converged_iter = zeros(trial_num,n_players)-1;
global_conv = 0;


for tr=1:trial_num
    for j=1:num_save_iters
        for pl=1:n_players
            for s=1:n_states
                for a=1:n_actions
                    l1_acc_pl(pl,j,tr) = l1_acc_pl(pl,j,tr)+abs(policy_iter(j,pl,a,s,tr)-...
                        policy_iter(end,pl,a,s,tr));
                    
                end
                polj = reshape(policy_iter(j,pl,:,s,tr),1,[]);
                polend = reshape(policy_iter(end,pl,:,s,tr),1,[]);
                if l1_acc_pl(pl,j,tr) < err && converged_iter(tr,pl) < 0
                    converged(tr,pl) = 1;
                    converged_iter(tr,pl) = j;
                end
            end
            l1_acc(j,tr) = l1_acc(j,tr)+l1_acc_pl(pl,j,tr);
        end
        l1_acc(j,tr) = l1_acc(j,tr)/n_players;
        if sum(sum(converged)) == n_players*n_states
            global_conv = 1;
        end
        
    end
end



l1_acc = zeros(num_save_iters,trial_num);
for tr=1:trial_num
    for j=1:num_save_iters
        for pl=1:n_players
            for s=1:n_states
                for a=1:n_actions
                    l1_acc(j,tr) = l1_acc(j,tr)+abs(policy_iter(j,pl,a,s,tr)-...
                        policy_iter(end,pl,a,s,tr));
                end
            end
        end
        l1_acc(j,tr) = l1_acc(j,tr)/n_players;
    end
end

l1_mean = mean(l1_acc,2);
l1_std = sqrt(var(l1_acc,0,2));

v_iter_mean = mean(v_iter,4);
v_iter_std = sqrt(var(v_iter,0,4));
q_iter_mean = mean(q_iter,5);
q_iter_std = sqrt(var(q_iter,0,5));




h = figure;
plot_ind = 1:length(iter_save);
x_label_ind = ceil(linspace(1,length(plot_ind),5));
% log
% x_vector = log([iter_save(plot_ind), fliplr(iter_save(plot_ind))]);
% LL_plot = log(iter_save(plot_ind));

x_vector = [iter_save(plot_ind), fliplr(iter_save(plot_ind))];
LL_plot = iter_save(plot_ind);

main_color = [1 0 1];
p_color = main_color+(1-main_color)*0.55;

set(gca,'XTick',LL_plot(x_label_ind),'XTickLabel',iter_save(x_label_ind));


hold on

data_err =  (l1_std(plot_ind))';
data_mean = (l1_mean(plot_ind))';
patch = fill(x_vector, [data_mean+data_err,fliplr(data_mean-data_err)],p_color);
set(patch, 'edgecolor', 'none');
set(patch, 'FaceAlpha', 0.3);
est = plot(LL_plot,data_mean,'Color',main_color,'linewidth',2);
target = yline(0,'--','Color','k','linewidth',2);
xlim([min(LL_plot) inf])
ylim([-max(l1_mean)-1 max(l1_mean)+1])
xlabel('Iterations')
ylabel('L1-accuracy')
set(gca,'FontSize',16)

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'res_dist1','-dpdf','-r0')


lw = 1.5;
win=10;
h=figure;
for i=1:trial_num
plot(LL_plot,movavg(l1_acc(:,i),'exponential',win),'linewidth',lw);
hold on
end

xlim([min(LL_plot) inf])
%ylim([-max(l1_mean)-1 max(l1_mean)+1])
xlabel('Iterations')
ylabel('L1-accuracy')
set(gca,'FontSize',16)

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'res_dist2','-dpdf','-r0')


toc
%% Saving
save result_distancing


