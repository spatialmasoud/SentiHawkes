%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compare SentHawkes prediction vs other state-of-the-art approaches
% (Creating tables of the results)
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

size_stations = 4;
D = 7;

RMSE_hawkes = zeros(size_stations, D,10);
RMSE_hawkes_feature2 = zeros(size_stations, D,10);
RMSE_lstm = zeros(size_stations, D,10);
RMSE_svm = zeros(size_stations, D,10);
RMSE_arima = zeros(size_stations, D,10);
RMSE_rmtpp = zeros(size_stations, D,10);

MAE_hawkes = zeros(size_stations, D,10);
MAE_hawkes_feature2 = zeros(size_stations, D,10);
MAE_lstm = zeros(size_stations, D,10);
MAE_svm = zeros(size_stations, D,10);
MAE_arima = zeros(size_stations, D,10);
MAE_rmtpp = zeros(size_stations, D,10);


for jjjj = 1:10

    load(strcat('workspaces\run', string(jjjj),'.mat'))
      
    for s=1:size_stations
        for c = 1:D
            RMSE_hawkes(s,c,jjjj) = sqrt(mse(count_ground_pred_hawkes{s,c}(1,:), count_ground_pred_hawkes{s,c}(2,:)));
            RMSE_hawkes_feature2(s,c,jjjj) = sqrt(mse(count_ground_pred_hawkes_with_features2{s,c}(1,:), count_ground_pred_hawkes_with_features2{s,c}(2,:)));
            RMSE_svm(s,c,jjjj) = sqrt(mse(count_ground_pred_svm{s,c}(1,:), count_ground_pred_svm{s,c}(2,:)));
            RMSE_arima(s,c,jjjj) = sqrt(mse(count_ground_pred_arima{s,c}(1,:), count_ground_pred_arima{s,c}(2,:)));
            RMSE_lstm(s,c,jjjj) = sqrt(mse(count_ground_pred_lstm{s,c}(1,:), count_ground_pred_lstm{s,c}(2,:)));
            RMSE_rmtpp(s,c,jjjj) = sqrt(mse(count_ground_pred_rmtpp{s,c}(1,:), count_ground_pred_rmtpp{s,c}(2,:)));
        end
    end
    
    for s=1:size_stations
        for c = 1:D
            MAE_hawkes(s,c,jjjj) = mae(count_ground_pred_hawkes{s,c}(1,:), count_ground_pred_hawkes{s,c}(2,:));
            MAE_hawkes_feature2(s,c,jjjj) = mae(count_ground_pred_hawkes_with_features2{s,c}(1,:), count_ground_pred_hawkes_with_features2{s,c}(2,:));
            MAE_svm(s,c,jjjj) = mae(count_ground_pred_svm{s,c}(1,:), count_ground_pred_svm{s,c}(2,:));
            MAE_arima(s,c,jjjj) = mae(count_ground_pred_arima{s,c}(1,:), count_ground_pred_arima{s,c}(2,:));
            MAE_lstm(s,c,jjjj) = mae(count_ground_pred_lstm{s,c}(1,:), count_ground_pred_lstm{s,c}(2,:));
            MAE_rmtpp(s,c,jjjj) = mae(count_ground_pred_rmtpp{s,c}(1,:), count_ground_pred_rmtpp{s,c}(2,:));
        end
    end

end

RMSE_hawkes_mean = mean(RMSE_hawkes,3);
RMSE_hawkes_std = std(RMSE_hawkes,[],3);
RMSE_hawkes_feature2_mean = mean(RMSE_hawkes_feature2 ,3);
RMSE_hawkes_feature2_std = std(RMSE_hawkes_feature2 ,[],3);
RMSE_lstm_mean = mean(RMSE_lstm,3);
RMSE_lstm_std = std(RMSE_lstm,[],3);
RMSE_svm_mean = mean(RMSE_svm,3);
RMSE_svm_std = std(RMSE_svm,[],3);
RMSE_arima_mean = mean(RMSE_arima, 3);
RMSE_arima_std = std(RMSE_arima,[],3);
RMSE_rmtpp_mean = mean(RMSE_rmtpp, 3);
RMSE_rmtpp_std = std(RMSE_rmtpp,[],3);

MAE_hawkes_mean = mean(MAE_hawkes,3);
MAE_hawkes_std = std(MAE_hawkes,[],3);
MAE_hawkes_feature2_mean = mean(MAE_hawkes_feature2 ,3);
MAE_hawkes_feature2_std = std(MAE_hawkes_feature2 ,[],3);
MAE_lstm_mean = mean(MAE_lstm,3);
MAE_lstm_std = std(MAE_lstm,[],3);
MAE_svm_mean = mean(MAE_svm,3);
MAE_svm_std = std(MAE_svm,[],3);
MAE_arima_mean = mean(MAE_arima, 3);
MAE_arima_std = std(MAE_arima,[],3);
MAE_rmtpp_mean = mean(MAE_rmtpp, 3);
MAE_rmtpp_std = std(MAE_rmtpp,[],3);


X = ["RMSE", 'MAE'];
Y = ["hawkes", 'hawkes_feature2', 'lstm', 'svm','arima','rmtpp'];

for x = 1:length(X)
    mat_mean = zeros(4*D,length(Y));
    mat_std = zeros(4*D,length(Y));
    metric = X(x);
    for y = 1:length(Y) %approach
        approach = Y(y);
        m=eval(strcat(metric,'_',approach,'_mean'));
        s=eval(strcat(metric,'_',approach,'_std'));
        col_m = reshape(m.',[4*7 1]);
        col_s = reshape(s.',[4*7 1]);
        mat_mean(:,y) = col_m;
        mat_std(:,y) = col_s; 
    end
    writematrix(mat_mean,'Results\main_mean.xls','sheet',metric)
    writematrix(mat_std,'Results\main_std.xls','sheet',metric)
end
disp('Excel files are successfully written to the results folder.')

