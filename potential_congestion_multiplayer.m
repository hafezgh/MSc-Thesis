clear all; close all; clc;
tic
rng(517);
trial_num = 4;

n_players = 8;
n_states = 2; % number of states (1 is safe and 2 is unsafe)
c=100*n_players; % penalty for being in the unsafe state
n_actions = 4; w = [1,2,4,6]; % number of actions per state (facilities) and their weights

num_iter = 3*10^9;
LL = space125(10,num_iter);
ll = length(LL);
gamma = 0.6;
rho_alpha = 0.9;
rho_beta = 1.0;
rho = 0.7;
tau_bar_zero = 0.068;
tau_bar_eps = 45000;
DD = 1/(1-gamma);
%tau0 = 2; %orig
tau0 = 3;
C=1.5;
eps_to = 0.0002;


%% Trials

v_iter = zeros(ll,n_players,n_states,trial_num);
q_iter = zeros(ll,n_players,n_actions,n_states,trial_num);

parfor trial = 1:trial_num

    % Initialization
    state = 1; % Initially at 
    tau = tau0.*ones(n_states,1);
    q = zeros(n_players,n_actions,n_states);
    v = zeros(n_players,n_states,1);
    v_iter_tmp = zeros(ll,n_players,n_states);
    q_iter_tmp = zeros(ll,n_players,n_actions,n_states);
    
    % Iterations
    state_counts=ones(1,n_states);
    
    
    
    for k = 1:num_iter
      
        if mod(k,10^5) == 0
            k/(10^5)
        end
        
        tau_old = tau(state);
        tau(state) = tau_to_zero_decay(state_counts(state), tau_bar_zero, rho_alpha, rho, DD);
        %tau(state) = tau_to_eps_decay(state_counts(state), tau_bar_eps, eps_to);
        actions = zeros(n_players,1);
        softmaxvs = zeros(n_players,1);
        a_probs = zeros(n_players,1);
        alpha_bars = zeros(n_players,1);
        
        for i=1:n_players
            [softmaxv, a, a_prob] = softmax_policy( q(i,:,state)',tau_old );
            actions(i)=find(a);
            softmaxvs(i) = softmaxv;
            a_probs(i) = a_prob;
            alpha_bars(i) = min(1,calc_learning_rate(state_counts(state),C,rho_alpha)/a_prob);
        end
        
        [r,next_state] = payoff_congestion_game(state,w,actions,c);

        for i=1:n_players
            q(i, actions(i),state) = q(i, actions(i),state)+ alpha_bars(i)*...
            (r(i) + gamma*v(i,next_state) - q(i, actions(i),state)); 
            v(i,state) = v(i,state) +...
                calc_learning_rate(state_counts(state),C,rho_beta)*(softmaxvs(i)-v(i,state));
        end
        
        if sum(find(LL==k))>0
            jj = find(LL==k);
            v_iter_tmp(jj,:,:) = v;
            q_iter_tmp(jj,:,:,:) = q;
        end
        
        state_counts(state)=state_counts(state)+1;
        state = next_state;

    end
    
    v_iter(:,:,:,trial) = v_iter_tmp;
    q_iter(:,:,:,:,trial) = q_iter_tmp;
    
end

%load results_new4

%% Post-processing and Plotting
plot_ind = 1:length(LL);
x_label_ind = ceil(linspace(1,length(plot_ind),5));
v_iter_mean = mean(v_iter,4);
v_iter_std = sqrt(var(v_iter,0,4));
q_iter_mean = mean(q_iter,5);
q_iter_std = sqrt(var(q_iter,0,5));

policies = zeros(n_players, n_states, n_actions);

temp = 1.0;
for i=1:n_players
    for s=1:n_states
        policies(i,s,:) = exp(q_iter_mean(end,i,:,s)/temp)./...
            (sum(exp(q_iter_mean(end,i,:,s)/temp)));
    end
end
policies

% Plotting to do
% figure
toc

%% Saving
save result_congestion_multiplayer


