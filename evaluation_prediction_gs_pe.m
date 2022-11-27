%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Compare SentHawkes prediction vs other state-of-the-art approaches
% (Creating figures of the results)
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

x_hawkes_all = cell(4,7);
y_hawkes_all = cell(4,7);
x_senthawkes_all = cell(4,7);
y_senthawkes_all = cell(4,7);
x_arima_all = cell(4,7);
y_arima_all = cell(4,7);
x_svm_all = cell(4,7);
y_svm_all = cell(4,7);
x_lstm_all = cell(4,7);
y_lstm_all = cell(4,7);
x_rmtpp_all =cell(4,7);
y_rmtpp_all =cell(4,7);

for jjjj = 1:10

    load(strcat('workspaces\run', string(jjjj),'.mat'))
    
    for s=1:size_stations
        for c = 1:D
            x_hawkes_all{s,c} = count_ground_pred_hawkes{s,c}(1,:);
            y_hawkes_all{s,c} = [y_hawkes_all{s,c}; count_ground_pred_hawkes{s,c}(2,:)];
            
            x_senthawkes_all{s,c} = count_ground_pred_hawkes_with_features2{s,c}(1,:);
            y_senthawkes_all{s,c} = [y_senthawkes_all{s,c}; count_ground_pred_hawkes_with_features2{s,c}(2,:)];
            
            x_arima_all{s,c} = count_ground_pred_arima{s,c}(1,:);
            y_arima_all{s,c} = [y_arima_all{s,c}; count_ground_pred_arima{s,c}(2,:)];
                        
            x_svm_all{s,c} = count_ground_pred_svm{s,c}(1,:);
            y_svm_all{s,c} = [y_svm_all{s,c}; count_ground_pred_svm{s,c}(2,:)];
            
            x_lstm_all{s,c} = count_ground_pred_lstm{s,c}(1,:);
            y_lstm_all{s,c} = [y_lstm_all{s,c}; count_ground_pred_lstm{s,c}(2,:)];
            
            x_rmtpp_all{s,c} = count_ground_pred_rmtpp{s,c}(1,:);
            y_rmtpp_all{s,c} = [y_rmtpp_all{s,c}; count_ground_pred_rmtpp{s,c}(2,:)];
        end
    end
end

figure
counter=1;
for s=1:size_stations
    for c = 1:D
        x_hawkes = x_hawkes_all{s,c};
        mean_y_hawkes = mean(y_hawkes_all{s,c},1);
        std_y_hawkes = std(y_hawkes_all{s,c},[],1);
        
        x_senthawkes = x_senthawkes_all{s,c};
        mean_y_senthawkes = mean(y_senthawkes_all{s,c},1);
        std_y_senthawkes = std(y_senthawkes_all{s,c},[],1);
        
        if max(x_hawkes)>0 && max(mean_y_hawkes)>0
            subplot(size_stations,D,counter)
            axis square
            hold on
            mx_hawks = max([max(x_hawkes),max(mean_y_hawkes)]);
            y_line = 0:1:mx_hawks;
            x_line = 0:1:mx_hawks;
            xlim([0 mx_hawks])
            ylim([0 mx_hawks])
            xlabel('Ground Truth')
            ylabel('Predicted Events')
            s1  = shadedErrorBar(x_hawkes, y_hawkes_all{s,c}, {@mean,@std}, 'lineprops','-r','transparent',true,'patchSaturation',0.05);
            s1.mainLine.LineWidth = 4;
            s2 = shadedErrorBar(x_senthawkes, y_senthawkes_all{s,c}, {@mean,@std}, 'lineprops', '-g','transparent',true,'patchSaturation',0.05);
            s2.mainLine.LineWidth = 4;
            line(x_line,y_line,'Color',[0 0 0])
            title(strcat('Station ',string(s),'- Class ',string(c)))
        end
        counter = counter+1;
    end
    legend('Hawkes','SentHawkes');
end

for s=1:4
    figure('units','normalized','outerposition',[0 0 1 1])
    name = ["Safety",'View','Information','Reliability','Comfort','Personnel','Additional'];
    t = tiledlayout(2,4,'TileSpacing','Compact','Padding','Compact');
    counter=1;
    for c = 1:D
        x_hawkes = x_hawkes_all{s,c};
        mean_y_hawkes = mean(y_hawkes_all{s,c},1);

        x_senthawkes = x_senthawkes_all{s,c};
        x_arima = x_arima_all{s,c};

        if max(x_hawkes)>0 && max(mean_y_hawkes)>0
            if counter ==3
                ax = nexttile;
            else
                nexttile;
            end
            axis square
            hold on
            mx_hawks = max([max(x_hawkes),max(mean_y_hawkes)]);
            y_line = 0:1:max(mean_y_hawkes);
            x_line = 0:1:max(x_hawkes);
            if mx_hawks==0
                mx_hawks = 1;
            end
            xlim([0 mx_hawks])
            ylim([0 mx_hawks])
            xlabel('Ground Truth')
            ylabel('Predicted Events')

            s1  = shadedErrorBar(x_hawkes, y_hawkes_all{s,c}, {@mean,@std}, 'lineprops','-r','transparent',true,'patchSaturation',0.05);
            s1.mainLine.LineWidth = 4;

            s2 = shadedErrorBar(x_senthawkes, y_senthawkes_all{s,c}, {@mean,@std}, 'lineprops', '-g','transparent',true,'patchSaturation',0.05);
            s2.mainLine.LineWidth = 4;

            s3 = shadedErrorBar(x_arima, y_arima_all{s,c}, {@mean,@std}, 'lineprops', '-b','transparent',true,'patchSaturation',0.05);
            s3.mainLine.LineWidth = 4;

            line(x_line,x_line,'Color',[0 0 0])
            grid on
            ax = gca;
            ax.FontSize = 14;
            ax.FontWeight = 'bold';
            title('title')
            legend({'MHP','SentHawkes', 'ARIMA'}, 'Orientation','horizontal', 'Location','Best');
            exportgraphics(ax,strcat('results\',string(s),'_',name(counter),'.pdf'),'ContentType','vector', 'Resolution',300)
        end
        counter = counter+1;
    end
end
