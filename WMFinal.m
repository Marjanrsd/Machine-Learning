clear; close all; clc
N1=4; % V1 neurons population
N2=4; % PFC neurons population
%% Set up the simulation parameters 
dt = 0.0005;                % Time step to use
tmax = 1;                   % Maximum time for simulation
tvec = 0:dt:tmax;           % Vector of time points
Nt = length(tvec);          % Number of time points

%% v1 neurons parameters
s = -40:1:40; %directions/degrees
v1_rmax = 52.14; % maximum firing rate (Hz)
smax = 0; % degrees 
x=s-smax;

sig_f = 14.73; % degrees 
% vector of firing rates for v1 neurons per stimulus
v1_rates_1 = v1_rmax .* exp(-0.5 .* (x / sig_f) .^ 2);

smax = 90; 
x=s-smax;
v1_rates_2 = v1_rmax .* exp(-0.5 .* (x / sig_f) .^ 2);

smax = 180; 
x=s-smax;
v1_rates_3 = v1_rmax .* exp(-0.5 .* (x / sig_f) .^ 2);

smax = 270; 
x=s-smax;
v1_rates_4 = v1_rmax .* exp(-0.5 .* (x / sig_f) .^ 2);

%% PFC neurons parameters
tar=0.01;
PFC_current_rate = zeros(N2, length(tvec));
% initial firing rate for pfc neurons
PFC_current_rate(1) = 2;
rmax=100;

W=5.5; % recurrent weights for PFC neurons
% v1->PFC weights
w = zeros(N1,N2);
w(1,1)=2.2;
w(1,2)=0.1;
w(1,3)=0.1;
w(1,4)=0.1;
w(2,1)=0.2;
w(2,2)=2.2;
w(2,3)=0.2;
w(2,4)=0.1;
w(3,1)=0.3;
w(3,2)=0.01;
w(3,3)=2.2;
w(3,4)=0.1;
w(4,1)=0.1;
w(4,2)=0.01;
w(4,3)=0.2;
w(4,4)=2.2;

rates1=v1_rates_1(41);
rates2=v1_rates_2(41);
rates3=v1_rates_3(41);
rates4=v1_rates_4(41);

% 4 main experiments, each with different directions
for exp_=1:4
    if exp_ == 2
        rates1=v1_rates_4(41);
        rates2=v1_rates_1(41);
        rates3=v1_rates_2(41);
        rates4=v1_rates_3(41);
    elseif exp_ == 3
        rates1=v1_rates_3(41);
        rates2=v1_rates_4(41);
        rates3=v1_rates_1(41);
        rates4=v1_rates_2(41);
    elseif exp_ == 4
        rates1=v1_rates_2(41);
        rates2=v1_rates_3(41);
        rates3=v1_rates_4(41);
        rates4=v1_rates_1(41);
    end
    for i=1:length(tvec)
        for j=1:N2
            % prevent matlab indexing error
            prev_index = max(1, i-1);
            Rnorm=PFC_current_rate(j,prev_index)./rmax;
            if i >= 1000 && i <= 1080
                total_input = sum(W.*Rnorm + w(1,j).*rates1./v1_rmax ... 
                                           + w(2,j).*rates2./v1_rmax ... 
                                           + w(3,j).*rates3./v1_rmax ... 
                                           + w(4,j).*rates4./v1_rmax);
            else
                total_input= sum(W.*Rnorm);
            end
            activation_output = rmax./(1 + exp(-(total_input-2.9)));
            dr_dt = (activation_output - PFC_current_rate(j,prev_index)) / tar;
            PFC_current_rate(j,i) = PFC_current_rate(j,prev_index) + dr_dt' * dt;       
        end
    end

    figure()
    subplot (2,2,1)
    plot(tvec, PFC_current_rate(1,:), LineWidth=2)
    title("Exp #" + exp_ + " PFC neuron 1")
    xlabel("Time (sec)")
    ylabel("Firing rates (HZ)")
    %figure()
    subplot (2,2,2)
    plot(tvec, PFC_current_rate(2,:), LineWidth=2)
    title("Exp #" + exp_ + " PFC neuron 2")
    xlabel("Time (sec)")
    ylabel("Firing rates (HZ)")
    %figure()
    subplot (2,2,3)
    plot(tvec, PFC_current_rate(3,:), LineWidth=2)
    title("Exp #" + exp_ + " PFC neuron 3")
    xlabel("Time (sec)")
    ylabel("Firing rates (HZ)")
    %figure()
    subplot (2,2,4)
    plot(tvec, PFC_current_rate(4,:),LineWidth=2)
    title("Exp #" + exp_ + " PFC neuron 4")
    xlabel("Time (sec)")
    ylabel("Firing rates (HZ)")
    pause()
    close all;
end