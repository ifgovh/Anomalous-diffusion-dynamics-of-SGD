clear
close all
% load data
d = dir('*net*epoch*');
[correlation_all,lag_all] = deal(cell(length(d),1));

% displacement distribution
dis_interval = 2.^(0:5);
for ii = 1:length(d)
    try
    [correlation_all{ii},lag_all{ii}] = get_v_acc(d,ii,dis_interval);
    catch
        correlation_all{ii} = nan;
        lag_all{ii} = nan;
    end
end

save('velocity_acc_2.mat','correlation_all','lag_all','-v7.3')

% plot
net_index = 4;
interval_index = 1;
loglog(lag_all{net_index}{interval_index},correlation_all{net_index}{interval_index})

function [correlation,lag] = get_v_acc(d,net_index,dis_interval)
datax_dir = dir(fullfile(d(net_index).folder,d(net_index).name,'*data_part*'));
for part = 1:length(datax_dir)
    L = load(fullfile(datax_dir(1).folder,[d(net_index).name(1:end-24),'_data_part_',num2str(part),'.mat']),'Displacement_all');
    try
        Displacement_all{part} = L.Displacement_all;
    catch
    end
end
for jj = 1:length(dis_interval)
    eval(['displacement_',num2str(dis_interval(jj)),' = [];']);
end

for epoch_seg = 1:length(Displacement_all)
    for jj = 1:length(dis_interval)
        %eval(['displacement_',num2str(dis_interval(jj)),' = diag(Displacement_all{',num2str(epoch_seg),'},',num2str(dis_interval(jj)),');']);
        try
            eval(['displacement_',num2str(dis_interval(jj)),' = [displacement_',num2str(dis_interval(jj)),';diag(Displacement_all{',num2str(epoch_seg),'},',num2str(dis_interval(jj)),')];']);
        catch
        end
    end
end
[correlation,lag] = deal(cell(length(dis_interval),1));
for jj = 1:length(dis_interval)
    eval(['all_disp = displacement_',num2str(dis_interval(jj)),';'])
    all_disp = all_disp.^0.5;
    [correlation{jj},lag{jj}] = autocorr(double(all_disp),'numlags',length(all_disp)-1);
end
end
