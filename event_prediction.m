%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Event prediction by SentHawkes vs other baseline approaches
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('BasicFunc');
addpath('Data');
addpath('Simulation');
addpath('Learning');
addpath('Analysis');
addpath('Visualization');

clear
close all
clc

for run_id=1:10
    clearvars -except run_id
    load('.\\Data\\Data_Sequence.mat')
    options.dt = 1; % time granularity
    options.Tmax = 5464; % the maximum size of time window (for training)
    options.M = round(options.Tmax./options.dt);

    D = 7; % the dimension of Hawkes processes
    nTest = 1; % iterations
    Nout = 20; % epochs <--> outer

    para.mu = rand(D,1)/D;
    para.A = rand(D, D);
    para.A = 0.65 * para.A./max(abs(eig(para.A)));
    para.A = reshape(para.A, [D, 1, D]);
    para.w = 1;

    [tmp, size_stations] = size(Seqs);

    models_Hawkes=cell(1,size_stations);
    models_sentHawkes=cell(1,size_stations);
    
    disp('###########################')
    disp(strcat('Run #',string(run_id)))
    disp('###########################')
    
    for s=1:size_stations
        disp(strcat('Learning Hawkes for station: ',string(s)))

        for n = 1:nTest
            % initialize
            model.A = rand(D,1,D)./(D^2);
            model.mu = rand(D,1)./D;
            model.s = rand(D,1,D)./(D^2);
            model.kernel = 'exp';
            model.w = 1;
            model.ws = 1;
            model.landmark = 0;

            disp('Learning SentHawkes')
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

            model_sentHawkes = Learning_MLE_Basis_Feature(Seqs(s), model, alg1);
            model_sentHawkes.Sentiment = 1;
            
            disp('Learning Hawkes')
            model_Hawkes = Learning_MLE_Basis(Seqs(s), model, alg1);
            model_Hawkes.Sentiment = 0;

        end

        models_Hawkes{1,s} = model_Hawkes;
        models_sentHawkes{1,s} = model_sentHawkes;

    end

    %% Prediction -  Hawkes

    options.NumTest = 1;
    options.tstep = 1;
    predicted_events = cell(1,size_stations);
    count_expected = cell(1,size_stations);

    for s=1:size_stations
        disp(strcat('Predicting events using Hawkes for Station ',string(s)))
        model = models_Hawkes{1,s};
        History_All = [Seqs(s).Time;Seqs(s).Mark];
        ground_truth_ind = find(History_All(1,:) > options.Tmax);
        History_test = History_All(:,ground_truth_ind);
        model.History_test = History_test;
        ind = find(History_All(1,:) < options.Tmax);
        History_train = History_All(:,ind);
        options.Nmax = 1000+length(History_train); % the maximum number of events per sequence
        [lambda_current, count_expect, pred_events] = Prediction_HP(options.Tmax,max([Seqs.Stop]), History_train, model, options);
        predicted_events{1,s} = pred_events;
        count_expected{1,s} = count_expect;
    end
    save(strcat('workspaces\run',string(run_id),'.mat'))

    %% Prediction - SentHawkes

    options.NumTest = 1;
    options.tstep = 1;
    predicted_events3 = cell(1,size_stations);
    count_expected3 = cell(1,size_stations);

    for s=1:size_stations
        disp(strcat('Predicting events using SentHawkes for Station ',string(s)))
        model = models_sentHawkes{1,s};
        History_All = [Seqs(s).Time;Seqs(s).Mark; Seqs(s).Sentiment];
        ground_truth_ind = find(History_All(1,:) > options.Tmax);
        History_test = History_All(:,ground_truth_ind);
        model.History_test = History_test;
        ind = find(History_All(1,:) < options.Tmax);
        History_train = History_All(:,ind);
        options.Nmax = 2000+length(History_train); % the maximum number of events per sequence

        [lambda_current, count_expect, pred_events] = Prediction_HP(options.Tmax,max([Seqs.Stop]), History_train, model, options);

        predicted_events3{1,s} = pred_events;
        count_expected3{1,s} = count_expect;
    end

    save(strcat('workspaces\run',string(run_id),'.mat'))

    %% Baseline approaches
    % continuous event sequence to discrete time-series
    
    ts_hourly = cell(size_stations,D);
    for s=1:size_stations
        History_All = [Seqs(s).Time;Seqs(s).Mark];
        for c = 1:D
            for t = 2:1:max([Seqs(s).Stop])
                ind = find(History_All(1,:) >= t-1 & History_All(1,:) <t & History_All(2,:)==c);
                ts_hourly{s,c} = [ts_hourly{s,c};[t,length(ind)]];
            end
        end
    end

    save(strcat('workspaces\run',string(run_id),'.mat'))

    %% LSTM - TS

    lstm_predicted_values = cell(size_stations,D);
    lstm_ground_values = cell(size_stations, D);
    for s=1:size_stations
        for d=1:D
            data = ts_hourly{s,d};
            data_train = data(1:floor(options.Tmax/1),2).';
            data_test = data(floor(options.Tmax/1)+1:end,2).';

            mu = mean(data_train);
            sig = std(data_train);

            dataTrainStandardized = (data_train - mu) / sig;
            XTrain = dataTrainStandardized(1:end-1);
            YTrain = dataTrainStandardized(2:end);

            numFeatures = 1;
            numResponses = 1;
            numHiddenUnits = 50;

            layers = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(numHiddenUnits)
                fullyConnectedLayer(numResponses)
                regressionLayer];

            options_lstm = trainingOptions('adam', ...
                'MaxEpochs',5, ...
                'GradientThreshold',1, ...
                'InitialLearnRate',0.005, ...
                'LearnRateSchedule','piecewise', ...
                'LearnRateDropPeriod',125, ...
                'LearnRateDropFactor',0.2, ...
                'Verbose',1, ...
                'Plots','none');

            net = trainNetwork(XTrain,YTrain,layers,options_lstm);

            dataTestStandardized = (data_test - mu) / sig;
            XTest = dataTestStandardized(1:end-1);

            net = predictAndUpdateState(net,XTrain);
            [net,YPred] = predictAndUpdateState(net,YTrain(end));

            numTimeStepsTest = numel(XTest);
            for i = 2:numTimeStepsTest
                [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
            end

            YPred = sig*YPred + mu;
            YTest = data_test(2:end);

            timestamps = Seqs(s).Time;
            timestamps_test_ids = timestamps>options.Tmax;
            timestamps_test = timestamps(timestamps_test_ids);
            YPred_with_t = [(1:1:length(YPred))+options.Tmax;YPred];

            for i = 1:length(timestamps_test)
                t = timestamps_test(i);
                ids = find(YPred_with_t(1,:)<t);
                ss = sum(YPred_with_t(2,ids));
                if ss<0 || isnan(ss) 
                    ss = 0;
                end
                lstm_predicted_values{s,d} = [lstm_predicted_values{s,d},[t;floor(ss)]];
            end

            History_All = [Seqs(s).Time;Seqs(s).Mark];
            ground_truth_ind = find(History_All(1,:) > options.Tmax & History_All(2,:)==d);
            History_test = History_All(:,ground_truth_ind);

            for i = 1:length(timestamps_test)
                t = timestamps_test(i);
                ids = find(History_test(1,:)<t);
                ss = sum(History_test(2,ids));
                if ss<0 || isnan(ss) 
                    ss = 0;
                end
                lstm_ground_values{s,d} = [lstm_ground_values{s,d},[t;floor(ss)]];
            end
        end
    end

    save(strcat('workspaces\run',string(run_id),'.mat'))



    %% SVR Regression
    svm_predicted_values = cell(size_stations,D);

    for s=1:size_stations
        for d=1:D
            data = ts_hourly{s,d};
            data_train = data(1:floor(options.Tmax/1),:);
            data_test = data(floor(options.Tmax/1)+1:end,:);

            Mdl = fitrsvm(data_train(:,1),data_train(:,2),'KernelFunction','gaussian','KernelScale','auto','Standardize',true);
            conv = Mdl.ConvergenceInfo.Converged;
            iter = Mdl.NumIterations;
            YPred = predict(Mdl, data_test(:,1));

            YPred_with_t = [(1:1:length(YPred))+options.Tmax;YPred.'];

            timestamps = Seqs(s).Time;
            timestamps_test_ids = timestamps>options.Tmax;
            timestamps_test = timestamps(timestamps_test_ids);

            for i = 1:length(timestamps_test)
                t = timestamps_test(i);
                ids = find(YPred_with_t(1,:)<t);
                ss = sum(YPred_with_t(2,ids));
                if ss<0 || isnan(ss) 
                    ss = 0;
                end
                svm_predicted_values{s,d} = [svm_predicted_values{s,d},[t;floor(ss)]];
            end

        end
    end
    save(strcat('workspaces\run',string(run_id),'.mat'))

    %% ARIMA - TS

    arima_predicted_values = cell(size_stations,D);
    for s=1:size_stations
        for d=1:D
            data = ts_hourly{s,d};
            data_train = data(1:floor(options.Tmax/1),2);
            length_data_train = length(data_train);
            data_test = data(floor(options.Tmax/1)+1:end,2);

            try
                Mdl = arima(1,1,1);
                EstMdl = estimate(Mdl,data_train);
                [YPred,yMSE] = forecast(EstMdl,2800,data_train);
            catch exception
                YPred = zeros(2800,1);
            end       

            timestamps = Seqs(s).Time;
            timestamps_test_ids = timestamps>options.Tmax;
            timestamps_test = timestamps(timestamps_test_ids);
            YPred_with_t = [(1:1:length(YPred))+options.Tmax;YPred.'];

            for i = 1:length(timestamps_test)
                t = timestamps_test(i);
                ids = find(YPred_with_t(1,:)<=t);
                ss = sum(YPred_with_t(2,ids));
                if ss<0 || isnan(ss) 
                    ss = 0;
                end
                arima_predicted_values{s,d} = [arima_predicted_values{s,d},[t;round(ss)]];
            end
        end
    end


    %% Importing RMTPP's results

    rmtpp_predicted_values = cell(size_stations,D);

    for s=1:size_stations
        filename = strcat('data\rmtpp_results\out',string(s-1),'.csv');
        T = readtable(filename);
        T = T{:,:};
        for d=1:D
            rmtpp_predicted_values{s,d} = [[options.Tmax:1:max([Seqs.Stop])];T(:,d).'];
        end
    end

    %% Evaluation with Number of tweets
    %Check whether predicted final retweet counts follow the ground-truth retweet counts

    count_ground_pred_hawkes = cell(size_stations,D);
    count_ground_pred_hawkes_with_features2 = cell(size_stations,D);
    count_ground_pred_svm = cell(size_stations,D);
    count_ground_pred_lstm = cell(size_stations,D);
    count_ground_pred_arima = cell(size_stations,D);
    count_ground_pred_rmtpp = cell(size_stations,D);

    for t=options.Tmax:1:max([Seqs.Stop])
        for s = 1:size_stations
            History_All = [Seqs(s).Time;Seqs(s).Mark];
            ground_truth_ind = find(History_All(1,:) > options.Tmax);
            History_test = History_All(:,ground_truth_ind);

            for c=1:D
                count_events_ground_truth = length(find(History_test(1,:) <= t & History_test(2,:)==c));
                count_events_pred = length(find(predicted_events{1,s}(1,:)<=t & predicted_events{1,s}(2,:)==c));
                count_ground_pred_hawkes{s,c} = [count_ground_pred_hawkes{s,c}, [count_events_ground_truth;count_events_pred]];

                count_events_ground_truth = length(find(History_test(1,:) <= t & History_test(2,:)==c));
                count_events_pred = length(find(predicted_events3{1,s}(1,:)<=t & predicted_events3{1,s}(2,:)==c));
                count_ground_pred_hawkes_with_features2{s,c} = [count_ground_pred_hawkes_with_features2{s,c}, [count_events_ground_truth;count_events_pred]];

                indx = max(find(lstm_ground_values{s,c}(1,:)<= t));
                if isempty(indx)
                    indx = 1;
                end

                count_ground_pred_lstm{s,c} = [count_ground_pred_lstm{s,c}, [count_events_ground_truth;lstm_predicted_values{s,c}(2,indx)]];
                count_ground_pred_svm{s,c} = [count_ground_pred_svm{s,c}, [count_events_ground_truth;svm_predicted_values{s,c}(2,indx)]];
                count_ground_pred_arima{s,c} = [count_ground_pred_arima{s,c}, [count_events_ground_truth;arima_predicted_values{s,c}(2,indx)]]; 
                count_ground_pred_rmtpp{s,c} = [count_ground_pred_rmtpp{s,c}, [count_events_ground_truth;rmtpp_predicted_values{s,c}(2,indx)]]; 
            end
        end
    end
    save(strcat('workspaces\run',string(run_id),'.mat'))
    
end
