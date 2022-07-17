clear all; close all; clc;
%% Initialization
tic
rng(3);

trial_num = 1;

n_states = 1; % number of states
n_actions = 2; % number of actions per state

num_iter = 10^4*3;

interval = 50;
interval_print = 10^4;
num_save_iters = num_iter/(interval);
iter_save = 1:interval:num_iter;

gamma = 0.6;
rho_alpha = 0.9;
rho_beta = 1.0;
rho = 0.7;

tau_bar_zero = 0.5;
DD = 1/(1-gamma);
tau0 = 2;
C=1.5;

% Payoff matrices
R1 = zeros(n_actions,n_actions,n_states);
R2 = zeros(n_actions,n_actions,n_states);

% 2 actions
R1(1,1) = 0.2;
R1(1,2) = 0.4;
R1(2,1) = 0.5;
R1(2,2) = 0.2;

R2(1,1) = -0.3;
R2(1,2) = 0.1;
R2(2,1) = 0.2;
R2(2,2) = 0.1;

% transition probabilities
pp = cell(n_states,n_states);
for i = 1:n_states
    for j1 = 1:n_actions
        for j2 = 1:n_actions
            ppp = rand(n_states,1)*rand + 0.1*ones(n_states,1); ppp = ppp/sum(ppp);
            for ii = 1:n_states
                pp{ii,i}(j1,j2) = ppp(ii);
            end
        end
    end
end


%% Trials

v1_iter = zeros(num_save_iters,n_states,trial_num);
v2_iter = zeros(num_save_iters,n_states,trial_num);
q1_iter = zeros(num_save_iters,n_actions,n_states,trial_num);
q2_iter = zeros(num_save_iters,n_actions,n_states,trial_num);

policy1 = zeros(num_save_iters,n_actions,n_states,trial_num);
policy2 = zeros(num_save_iters,n_actions,n_states,trial_num);

for trial = 1:trial_num

    % Initialization
    state = randi(n_states,1);
    tau = tau0.*ones(n_states,1);
    q1 = rand(n_actions,n_states);
    q2 = rand(n_actions,n_states);
    v1 = zeros(n_states,1);
    v1_iter_tmp = zeros(num_save_iters,n_states);
    q1_iter_tmp = zeros(num_save_iters,n_actions,n_states);
    v2 = zeros(n_states,1);
    v2_iter_tmp = zeros(num_save_iters,n_states);
    q2_iter_tmp = zeros(num_save_iters,n_actions,n_states);

    policy1_tmp = zeros(num_save_iters,n_actions,n_states);
    policy2_tmp = zeros(num_save_iters,n_actions,n_states);
    
    % Iterations
    state_counts=ones(1,n_states);
    
    
    
    for k = 1:num_iter
      
        if mod(k,interval_print) == 0
            k/(interval_print)
            
        end
        
        tau_old = tau(state);
        tau(state) = tau_to_zero_decay(state_counts(state), tau_bar_zero, rho_alpha, rho, DD);

        [softmaxv1, a1, a1_prob, br1] = softmax_policy( q1(:,state),tau_old ); 
        [softmaxv2, a2, a2_prob, br2] = softmax_policy( q2(:,state),tau_old );

        i1=find(a1); i2=find(a2);
        
        if isnan(a1_prob) || isnan(a2_prob)
            break;
        end
        
        next_state = sel_next_state(pp,state,a1,a2);
        
        
        alpha_bar1 = min(1,calc_learning_rate(state_counts(state),C,rho_alpha)/a1_prob);
        alpha_bar2 = min(1,calc_learning_rate(state_counts(state),C,rho_alpha)/a2_prob);

        q1(i1,state) = q1(i1,state) + alpha_bar1*...
            (R1(i1,i2,state) + gamma*v1(next_state) - q1(i1,state)); 
        v1(state) = v1(state) + calc_learning_rate(state_counts(state),C,rho_beta)*(softmaxv1-v1(state));
        
        q2(i2,state) = q2(i2,state) + alpha_bar2*...
            (R2(i2,i1,state) + gamma*v2(next_state) - q2(i2,state)); 
        v2(state) = v2(state) + calc_learning_rate(state_counts(state),C,rho_beta)*(softmaxv2-v2(state));
        
       
        if sum(find(iter_save==k))>0
            jj = find(iter_save==k);
            v1_iter_tmp(jj,:) = v1;
            v2_iter_tmp(jj,:) = v2;
            q1_iter_tmp(jj,:,:) = q1;
            q2_iter_tmp(jj,:,:) = q2;
%             for s=1:n_states
%                 policy1_tmp(jj,:,s) = ;
%             end
        end
        
        state_counts(state)=state_counts(state)+1;
        state = next_state;

    end
    
    v1_iter(:,:,trial) = v1_iter_tmp;
    v2_iter(:,:,trial) = v2_iter_tmp;
    q1_iter(:,:,:,trial) = q1_iter_tmp;
    q2_iter(:,:,:,trial) = q2_iter_tmp;
    
end

%load results_new4

%% Plotting and post-processing
v1_iter_mean = mean(v1_iter,3);
v1_iter_std = sqrt(var(v1_iter,0,3));
v2_iter_mean = mean(v2_iter,3);
v2_iter_std = sqrt(var(v2_iter,0,3));
q1_iter_mean = mean(q1_iter,4);
q1_iter_std = sqrt(var(q1_iter,0,4));
q2_iter_mean = mean(q2_iter,4);
q2_iter_std = sqrt(var(q2_iter,0,4));

policy_p1 = zeros(n_states, n_actions);
policy_p2 = zeros(n_states, n_actions);



plot_ind = 1:num_save_iters;
x_label_ind = ceil(linspace(1,length(plot_ind),5));

h=figure;
x_vector = [iter_save(plot_ind), fliplr(iter_save(plot_ind))];
iter_save_plot = iter_save(plot_ind);
%set(gca,'XTick',iter_save_plot(x_label_ind),'XTickLabel',iter_save(x_label_ind));


hold on

C = {'k','b','r','g',[0.9290 0.6940 0.1250],[.5 .6 .7],[.8 .2 .6],[0.5 0.2 0.6]};

w = 12;
data1 = (q1_iter_mean(plot_ind,1));
data2 = (q2_iter_mean(plot_ind,1));
p1a1 = plot(iter_save_plot,movavg(data1,'exponential',w),'Color',C{1},'linewidth',2);
p2a1 = plot(iter_save_plot,movavg(data2,'exponential',w),'Color',C{n_actions+1},'linewidth',2);

data1 = (q1_iter(plot_ind,2));
data2 = (q2_iter(plot_ind,2));
p1a2 = plot(iter_save_plot,movavg(data1,'exponential',w),'Color',C{2},'linewidth',2);
p2a2 = plot(iter_save_plot,movavg(data2,'exponential',w),'Color',C{n_actions+2},'linewidth',2);


legend([p1a1,p2a1,p1a2,p2a2],...
    'p1a1','p2a1','p1a2','p2a2');
xlabel('Iterations')
ylabel('local Q values')
set(gca,'FontSize',16)

set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'pot2p2a','-dpdf','-r0')
toc

%% Saving
save results_pot_2p2a


