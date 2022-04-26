clear all; close all; clc;
%% Initialization
tic
rng(517);
trial_num = 4;

n_states = 5; % number of states
n_actions = 2; % number of actions per state
num_iter = 3*10^9;
LL = space125(10,num_iter);
ll = length(LL);
gamma = 0.6;
rho_alpha = 0.9;
rho_beta = 1.0;
rho = 0.7;
%tau_bar_zero = 0.068; %orig
%tau_bar_eps = 45000; %orig
tau_bar_zero = 0.068;
tau_bar_eps = 45000;
DD = 1/(1-gamma);
tau0 = 2;
C=1.5;
eps_to = 0.0002;


% Payoff matrices
R1 = zeros(n_actions,n_actions,n_states);
R2 = zeros(n_actions,n_actions,n_states);


R1(:,:,1) = [[-0.5,0.2];[0.3,0.15]];
R2(:,:,1) = [[-0.5,0.3];[0.2,0.15]];

R1(:,:,2) = [[0.4,0.15];[0.2,-0.1]];
R2(:,:,2) = [[0.4,0.2];[0.15,-0.1]];

R1(:,:,3) = [[0.1,0.65];[-0.45,0.25]];
R2(:,:,3) = [[0.1,-0.45];[0.65,0.25]];

R1(:,:,4) = [[0.3,0.75];[0.55,0.45]];
R2(:,:,4) = [[0.3,0.55];[0.75,0.45]];

R1(:,:,5) = [[-0.8,-0.65];[-0.55,-0.35]];
R2(:,:,5) = [[-0.8,-0.55];[-0.65,-0.35]];

% transition probabilities
pp = cell(n_states,n_states);
for i = 1:n_states
    for j1 = 1:n_actions
        for j2 = 1:n_actions
            ppp = rand(n_states,1)*rand + 0.1*ones(n_states,1); ppp = ppp/sum(ppp);
            %ppp = ones(n_states,1)*(1/n_states);
            for ii = 1:n_states
                pp{ii,i}(j1,j2) = ppp(ii);
            end
        end
    end
end


%% Value iteration
Sv1_prev = zeros(1,n_states);
Sv2_prev = zeros(1,n_states);
Sv1 = zeros(1,n_states);
Sv2 = zeros(1,n_states);
opt_policy_p1 = zeros(n_actions,n_states);
opt_policy_p2 = zeros(n_actions,n_states);
opt_policy_p1(1,1:2)=1; opt_policy_p2(1,1:2)=1;
opt_policy_p1(2,3)=1; opt_policy_p2(2,3)=1;
opt_policy_p1(2,4)=1; opt_policy_p2(1,4)=1;
opt_policy_p1(2,5)=1; opt_policy_p2(2,5)=1;


theta = 0.0000001;

for s=1:n_states
    sv1 = Sv1(s);
    sv1_prev = Sv1(s);
    sv2 = Sv2(s);
    sv2_prev = Sv2(s);
    t = 0;
    cur_s = s;
    while 1==1
        a1 = sample_from_prob(opt_policy_p1(:,cur_s));
        a2 = sample_from_prob(opt_policy_p2(:,cur_s));

        sv1 = sv1+(gamma^t)*R1(a1,a2,cur_s);
        sv2 = sv2+(gamma^t)*R2(a1,a2,cur_s);
        
        if abs(sv1-sv1_prev)<theta && abs(sv2-sv2_prev)<theta
            Sv1(s) = sv1;
            Sv2(s) = sv2;
            break
        end
        sv1_prev = sv1;
        sv2_prev = sv2;
        cur_s = sample_from_prob(1/n_states*ones(1,n_states));
        t = t+1;
    end
end



%% Trials

v1_iter = zeros(ll,n_states,trial_num);
v2_iter = zeros(ll,n_states,trial_num);
q1_iter = zeros(ll,n_actions,n_states,trial_num);
q2_iter = zeros(ll,n_actions,n_states,trial_num);
parfor trial = 1:trial_num

    % Initialization
    state = randi(n_states,1);
    tau = tau0.*ones(n_states,1);
    q1 = zeros(n_actions,n_states);
    v1 = zeros(n_states,1);
    v1_iter_tmp = zeros(ll,n_states);
    q1_iter_tmp = zeros(ll,n_actions,n_states);
    q2 = zeros(n_actions,n_states);
    v2 = zeros(n_states,1);
    v2_iter_tmp = zeros(ll,n_states);
    q2_iter_tmp = zeros(ll,n_actions,n_states);
    
    % Iterations
    state_counts=ones(1,n_states);
    
    
    
    for k = 1:num_iter
      
        if mod(k,10^6) == 0
            k/(10^6)
        end
        
        tau_old = tau(state);
        tau(state) = tau_to_zero_decay(state_counts(state), tau_bar_zero, rho_alpha, rho, DD);
        %tau(state) = tau_to_eps_decay(state_counts(state), tau_bar_eps, eps_to);

        [softmaxv1, a1, a1_prob] = softmax_policy( q1(:,state),tau_old ); 
        [softmaxv2, a2, a2_prob] = softmax_policy( q2(:,state),tau_old ); 
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
        
       
        if sum(find(LL==k))>0
            jj = find(LL==k);
            v1_iter_tmp(jj,:) = v1;
            v2_iter_tmp(jj,:) = v2;
            q1_iter_tmp(jj,:,:) = q1;
            q2_iter_tmp(jj,:,:) = q2;
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

%% Plotting
plot_ind = 1:length(LL);
x_label_ind = ceil(linspace(1,length(plot_ind),5));
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

temp = 1.0;
for s=1:n_states
    policy_p1(s,:) = exp(q1_iter_mean(end,:,s)/temp)./sum(exp(q1_iter_mean(end,:,s)/temp));
    policy_p2(s,:) = exp(q2_iter_mean(end,:,s)/temp)./sum(exp(q2_iter_mean(end,:,s)/temp));
end
policy_p1
policy_p2
figure
x_vector = log([LL(plot_ind), fliplr(LL(plot_ind))]);
LL_plot = log(LL(plot_ind));
main_color1 = [0.8500 0.3250 0.0980];
p1_color = main_color1+(1-main_color1)*0.55;
main_color2 = [0.4940 0.1840 0.5560];
p2_color = main_color2+(1-main_color2)*0.55;
set(gca,'XTick',LL_plot(x_label_ind),'XTickLabel',LL(x_label_ind));


hold on
for i = 1:n_states
    data_err_1 =  (v1_iter_std(plot_ind,i))';
    data_mean_1 = (v1_iter_mean(plot_ind,i))';
    patch_1 = fill(x_vector, [data_mean_1+data_err_1,fliplr(data_mean_1-data_err_1)], p1_color);
    set(patch_1, 'edgecolor', 'none');
    set(patch_1, 'FaceAlpha', 0.3);
    
    data_err_2 =  (v2_iter_std(plot_ind,i))';
    data_mean_2 = (v2_iter_mean(plot_ind,i))';
    patch_2 = fill(x_vector, [data_mean_2+data_err_2,fliplr(data_mean_2-data_err_2)], p2_color);
    set(patch_2, 'edgecolor', 'none');
    set(patch_2, 'FaceAlpha', 0.3);
    
    est1 = plot(LL_plot,data_mean_1,'Color',main_color1,'linewidth',2);
    est2 = plot(LL_plot,data_mean_2,'Color',main_color2,'linewidth',2);
    target1 = yline(Sv1(i),'--','Color',main_color1,'linewidth',2);
    target2 = yline(Sv2(i),'--','Color',main_color2,'linewidth',2);
    leg_subset = [est1 est2];
    legend(leg_subset,'player 1','player 2');
end
xlim([min(LL_plot) inf])
%ylim([0.2 1.5])
xlabel('Iterations')
ylabel('Value Functions')
set(gca,'FontSize',16)
toc

%% Saving
save results_pot_non_transitive_hard5


