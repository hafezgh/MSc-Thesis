clear all; close all; clc;
%% Initialization
tic
rng(518);

n_actions = 2; % number of actions
num_iter = 6*10^6;

num_save_iters = num_iter/(10^4);
iter_save = 0:10^4:num_iter;

gamma = 0.5;

tau = 0.01;
dt = 0.000001;


% Payoff matrices
R1 = zeros(n_actions,n_actions);
R2 = zeros(n_actions,n_actions);

% 2 actions
R1(1,1) = 0.2;
R1(1,2) = 0.4;
R1(2,1) = 0.5;
R1(2,2) = 0.2;

R2(1,1) = -0.3;
R2(1,2) = 0.1;
R2(2,1) = 0.2;
R2(2,2) = 0.1;



%% Iterations

v1 = 0.0;
v2 = 0;

Q1 = R1+gamma*v1;
Q2 = R2'+gamma*v2;

q1_iter = zeros(num_save_iters,n_actions);
q2_iter = zeros(num_save_iters,n_actions);


% Initialization
q1 = rand(n_actions,1);
q2 = rand(n_actions,1)-0.5;

counter_save = 1;

for k = 1:num_iter
  
    if mod(k,10^6) == 0
        k/(10^6)
    end
    
    if mod(k,10^4) == 0
        q1_iter(counter_save,:) = q1;
        q2_iter(counter_save,:) = q2;
        counter_save = counter_save + 1;
    end
    

    [softmaxv1, a1, a1_prob, br1] = softmax_policy( q1,tau ); 
    [softmaxv2, a2, a2_prob, br2] = softmax_policy( q2,tau ); 
    i1=find(a1); i2=find(a2);

    if isnan(a1_prob) || isnan(a2_prob)
        disp("too large logits!");
        break;
    end
   
    dq1 = (Q1*br2-q1)*dt;
    dq2 = (Q2*br1-q2)*dt;

    q1 = q1+dq1;
    q2 = q2+dq2;
    
end
    
 
%load results_new4

%% Plotting

plot_ind = 1:num_save_iters;
x_label_ind = ceil(linspace(1,length(plot_ind),5));

q1_iter_mean = mean(q1_iter,3);
q1_iter_std = sqrt(var(q1_iter,0,3));
q2_iter_mean = mean(q2_iter,3);
q2_iter_std = sqrt(var(q2_iter,0,3));


figure
x_vector = [iter_save(plot_ind), fliplr(iter_save(plot_ind))];
iter_save_plot = iter_save(plot_ind);
main_color1 = [0.8500 0.3250 0.0980];
p1_color = main_color1+(1-main_color1)*0.55;
main_color2 = [0.4940 0.1840 0.5560];
p2_color = main_color2+(1-main_color2)*0.55;
% set(gca,'XTick',iter_save_plot(x_label_ind),'XTickLabel',iter_save(x_label_ind));


hold on

C = {'k','b','r','g',[0.9290 0.6940 0.1250],[.5 .6 .7],[.8 .2 .6],[0.5 0.2 0.6]};


data1 = (q1_iter_mean(plot_ind,1))';
data2 = (q2_iter_mean(plot_ind,1))';
p1a1 = plot(iter_save_plot,data1,'Color',C{1},'linewidth',2);
p2a1 = plot(iter_save_plot,data2,'Color',C{n_actions+1},'linewidth',2);

data1 = (q1_iter_mean(plot_ind,2))';
data2 = (q2_iter_mean(plot_ind,2))';
p1a2 = plot(iter_save_plot,data1,'Color',C{2},'linewidth',2);
p2a2 = plot(iter_save_plot,data2,'Color',C{n_actions+2},'linewidth',2);

%2a
pot_idx1 = 2;
pot_idx2 = 1;



for i = 1:n_actions
    target1 = yline(Q1(i,pot_idx1),'--','Color',C{i},'linewidth',2);
    target2 = yline(Q2(i,pot_idx2),'--','Color',C{n_actions+i},'linewidth',2);
end

legend([p1a1,p2a1,p1a2,p2a2],'p1a1','p2a1','p1a2','p2a2');


xlim([min(iter_save_plot) inf])
ylim([-0.6,0.6])
%ylim([min(min(min(Q1,Q2)))-0.1 max(max(max(Q1,Q2)))+0.1])
xlabel('Iterations')
ylabel('local Q values')
set(gca,'FontSize',16)


toc

%% Saving
save results_ode_a2

