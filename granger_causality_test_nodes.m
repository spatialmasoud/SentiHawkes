%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Significance test for causal relationships for transport nodes
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('BasicFunc');
addpath('Data');
addpath('Simulation');
addpath('Learning');
addpath('Analysis');
addpath('Visualization');

clc 
clear
close all

load('.\\Data\\Data_Station_As_Label.mat')

options.dt = 1; % time granularity
options.Tmax = 5000; % the maximum size of time window (for training)
options.M = round(options.Tmax./options.dt);

D = 7; % the dimension of Hawkes processes
nTest = 1; % iterations
Nout = 20; % epochs <--> outer

n_sampling = 1000;
size_stations = 4;
models=cell(D,n_sampling);

for c=1:D
    for j = 1:n_sampling
        u=[];
        m = 5; %average_block_size
        T = length(Seqs(c).Time);
        u(1) = ceil(T*rand);
        for t=2:T
            if rand<1/m
                u(t) = ceil(T*rand);
            else
                u(t) = u(t-1) + 1;
            end
        end
        
        u = sort(u);
        seq = struct('Time', [], ...
                      'Mark', [], ...
                      'Start', [], ...
                      'Stop', [], ...
                      'Sentiment', [], ...
                      'Location', []);

        seq_time_repl = [Seqs(c).Time;Seqs(c).Time];
        seq.Time = seq_time_repl(u);
        
        seq_mark_repl = [Seqs(c).Mark;Seqs(c).Mark];
        seq.Mark = seq_mark_repl(u);
        
        seq_sentiment_repl = [Seqs(c).Sentiment;Seqs(c).Sentiment];
        seq.Sentiment = seq_sentiment_repl(u);
        
        seq.Stop = Seqs(c).Stop;
        seq.Start = Seqs(c).Start;
        
        para.mu = rand(size_stations,1)/size_stations;
        para.A = rand(size_stations, size_stations);
        para.A = 0.65 * para.A./max(abs(eig(para.A)));
        para.A = reshape(para.A, [size_stations, 1, size_stations]);
        para.w = 1;

        for n = 1:nTest
            % initialize
            model.A = rand(size_stations,1,size_stations)./(size_stations^2);
            model.mu = rand(size_stations,1)./size_stations;
            model.s = rand(size_stations,1,size_stations)./(size_stations^2);
            model.kernel = 'exp';
            model.w = 1;
            model.landmark = 0;
            alg1.LowRank = 0;
            alg1.Sparse = 1;
            alg1.GroupSparse = 0;
            alg1.alphaS = 10;
            alg1.alphaG = 100; 
            alg1.alphaP = 1000; 
            alg1.outer = Nout;
            alg1.rho = 0.1;
            alg1.inner = 1;
            alg1.thres = 1e-5;
            alg1.Tmax = [];
            alg1.storeErr = 0;
            alg1.storeLL = 0;
            alg1.truth = para;
            model = Learning_MLE_Basis_Feature(seq, model, alg1);
        end
        models{c,j} = model; 
        save('workspaces\significance_test_causal_nodes.mat')
    end
end

%% Testing
load('workspaces\significance_test_causal_nodes.mat')
A_out = cell(D,1);
A_out_tested = cell(D,1);
dispersion_out = cell(D,1);
mean_value_out = cell(D,1);
confidence_intervals_out_max = cell(D,1);
confidence_intervals_out_min = cell(D,1);

for c=1:D
    events_number = length(Seqs(c).Time);
    A = zeros(size_stations,size_stations,n_sampling);
    for j = 1:n_sampling
        model = models{c,j};
        [A1, Phi1] = ImpactFunc(model, options);
        A(:,:,j) = squeeze(sum(Phi1,2));
    end
    
    A_out{c,1} = A;
    dispersion_out{c,1} = nanstd(A,[],3);
    mean_value_out{c,1} = nanmean(A,3);
    confidence_intervals_out_max{c,1} = mean_value_out{c,1} + 1.96*dispersion_out{c,1}/sqrt(n_sampling);
    confidence_intervals_out_min{c,1} = mean_value_out{c,1} - 1.96*dispersion_out{c,1}/sqrt(n_sampling);
    
    A_out_tested{c,1} = zeros(size_stations,size_stations);
    for i=1:size_stations
        for j=1:size_stations
            values = squeeze(A(i,j,:));
            sd_values = nanstd(values,[],1);
            test_array = values-1.96*sd_values/sqrt(n_sampling);
            if (isempty(test_array(test_array<=0)))
                A_out_tested{c,1}(i,j) = 1;
            end
        end
    end
end
 
%% Comparing causal graphs
figure
counter = 1;
X = categorical({'Safety','View','Information','Reliability','Comfort','Personnel','Additional'});
X = reordercats(X,{'Safety','View','Information','Reliability','Comfort','Personnel','Additional'});
for c = 1:D
    subplot(3,3,counter)
    imagesc(mean_value_out{c,1})
    caxis([0 1]);
    colorbar;
    title(X(c))
    axis square
    counter=counter+1;
end

%% with test

figure
counter = 1;
X = categorical({'Safety','View','Information','Reliability','Comfort','Personnel','Additional'});
X = reordercats(X,{'Safety','View','Information','Reliability','Comfort','Personnel','Additional'});
for c = 1:D
    subplot(2,4,counter)
    imagesc((mean_value_out{c,1} .* A_out_tested{c,1})>0,[0,1])
    caxis([0 1]);
    title(X(c))
    axis square
    counter=counter+1;
end