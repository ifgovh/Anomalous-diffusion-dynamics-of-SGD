% calculate the statistical property of SGD walk
function SGD_analysis_step_level(varargin)
% read in data
d = dir('resnet14*');
calculate = 0;
steps_in_part = 1e3;
% Loop number for PBS array job
loop_num = 0;

for ii = 1:length(d)
    %GN_dir = dir(fullfile(d(ii).folder,d(ii).name,'*gradient_noise*mat'));
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*sub_loss_w.mat'));
    % For PBS array job
    loop_num = loop_num + 1;
    if nargin ~= 0
        PBS_ARRAYID = varargin{1};
        if loop_num ~=  PBS_ARRAYID
            continue;
        end
    end
    
    %     if isempty(GN_dir)
    %         warning([d(ii).name, ' is not completed!'])
    %         continue
    %     end
    if calculate
        part_num = 1;
        index = 1;
        for file = 1:length(sub_loss_w_dir)
            R = load(fullfile(d(ii).folder,d(ii).name,sprintf('model_%d_sub_loss_w.mat',file)));
            % organise the weights
            all_weights_tmp = cellfun(@(x) reshape(x,1,[]) ,R.sub_weights,'UniformOutput',false);
            all_sub_weights{index} = cell2mat(all_weights_tmp);
            all_sub_loss{index} = R.sub_loss;
            all_test_sub_loss{index} = R.test_sub_loss;
            % if the variable is too large, split it.
            % attrib = whos('all_sub_weights');
            %if attrib.bytes/1024^3 > 5
            % use trajectory length to split the data
            if index*numel(R.sub_loss) > steps_in_part || file == length(sub_loss_w_dir)
                MSD_feed = organize_data(all_sub_weights, all_sub_loss, all_test_sub_loss, d(ii), sub_loss_w_dir(1), part_num);
                clear all_sub_weights
                calculate_MSD_trajectory(MSD_feed, all_sub_loss, all_test_sub_loss, d(ii), sub_loss_w_dir(1), part_num)
                clear all_sub_loss all_test_sub_loss
                part_num = part_num + 1;
                
                index = 1; % start again
            end
            index = index + 1;
        end
        
    end
    
    % plot
    %sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*sub_loss_w.mat'));
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*.t7'));
    datax_dir = dir(fullfile(sub_loss_w_dir(1).folder,'*data_part*'));
    
    for part = 1:length(datax_dir)-1
        try
            L = load(fullfile(sub_loss_w_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'loss','test_loss','delta_train_loss','delta_test_loss',...
                'MSD','Mean_contourlength','tau','MSD_forcontour','MSL_contourlength','contour_length','MSL_distance',...
                'distance','Displacement_all','Contour_length_all','f_loss','P1_loss','f_test_loss','P1_test_loss');
            
            loss{part} = L.loss;
            test_loss{part} = L.test_loss;
            delta_train_loss{part} = L.delta_train_loss;
            delta_test_loss{part} = L.delta_test_loss;
            MSD{part} = L.MSD;
            Mean_contourlength{part} = L.Mean_contourlength;
            tau{part} = L.tau;
            MSD_forcontour{part} = L.MSD_forcontour;
            MSL_contourlength{part} = L.MSL_contourlength;
            contour_length{part} = L.contour_length;
            MSL_distance{part} = L.MSL_distance;
            distance{part} = L.distance;
            Displacement_all{part} = L.Displacement_all;
            Contour_length_all{part} = L.Contour_length_all;
            f_loss{part} = L.f_loss;
            P1_loss{part} = L.P1_loss;
            f_test_loss{part} = L.f_test_loss;
            P1_test_loss{part} = L.P1_test_loss;
        catch
            loss{part} = nan;
            test_loss{part} = nan;
            delta_train_loss{part} = nan;
            delta_test_loss{part} = nan;
            MSD{part} = nan;
            Mean_contourlength{part} = nan;
            tau{part} = nan;
            MSD_forcontour{part} = nan;
            MSL_contourlength{part} = nan;
            contour_length{part} = nan;
            MSL_distance{part} = nan;
            distance{part} = nan;
            Displacement_all{part} = nan;
            Contour_length_all{part} = nan;
            f_loss{part} = nan;
            P1_loss{part} = nan;
            f_test_loss{part} = nan;
            P1_test_loss{part} = nan;
        end        
    end
    % starting moment of square displacement
    t_start = [2.^(0:8),400];
    figure_width = 25;
    total_row = 2;
    total_column = 2;
    
    % [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
    EMH = 0.2;
    EMV = 0.4;
                            

    % displacement distribution
    dis_interval = 2.^(0:8);    
    for jj = 1:length(dis_interval)
            eval(['displacement_',num2str(dis_interval(jj)),' = [];']);
    end
    % for distribution
    for epoch_seg = 1:length(tau)
        for jj = 1:length(dis_interval)
            %eval(['displacement_',num2str(dis_interval(jj)),' = diag(Displacement_all{',num2str(epoch_seg),'},',num2str(dis_interval(jj)),');']);
            try
                eval(['displacement_',num2str(dis_interval(jj)),' = [displacement_',num2str(dis_interval(jj)),';diag(Displacement_all{',num2str(epoch_seg),'},',num2str(dis_interval(jj)),')];']);
            catch
            end
        end
    end
    
    % for rescale
    for epoch_seg = 1:length(tau)
        for jj = 1:length(dis_interval)
            try
                eval(['displacement_scale{',num2str(epoch_seg),',',num2str(jj),'} = sqrt(diag(Displacement_all{',num2str(epoch_seg),'},',num2str(dis_interval(jj)),')./MSD{',num2str(epoch_seg),'}(',num2str(dis_interval(jj)),'));']);
            catch
            end
        end
    end
    
    close all
    [ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis( total_row, total_column, figure_width, EMH, 0.4, EMV, 0.4, 0.68, 0.7,5,4);
    % uniform FontSize and linewidth
    figure('NumberTitle','off','name', 'displacement_distri', 'units', 'centimeters', ...
        'color','w', 'position', [0, 0, figure_width, figure_hight], ...
        'PaperSize', [figure_width, figure_hight]); % this is the trick!
    
    subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(dis_interval));
    for jj = 1:length(dis_interval)
        eval(['all_disp = displacement_',num2str(dis_interval(jj)),';'])
        all_disp = all_disp.^0.5;
        log_space_min = log10(min(all_disp));
        if isinf(log_space_min)
            temp = unique(all_disp);
            log_space_min = log10(temp(2));
        end
        [N,edges] = histcounts(all_disp,logspace(log_space_min,log10(max(all_disp)),51),'Normalization','probability');
        plot(edges(1:end-1),N,'.-','color',map(jj,:))
    end
    for jj = 1:length(dis_interval)
        legend_string{jj} = ['\tau = ',num2str(dis_interval(jj))];
    end
    legend(legend_string)
    set(gca,'xscale','log','yscale','log')
    xlabel('Displacement')
    ylabel('PDF')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    for jj = 1:length(dis_interval)
        eval(['all_disp = displacement_',num2str(dis_interval(jj)),';'])
        log_space_min = log10(min(all_disp));
        if isinf(log_space_min)
            temp = unique(all_disp);
            log_space_min = log10(temp(2));
        end
        [N,edges] = histcounts(all_disp,logspace(log_space_min,log10(max(all_disp)),51),'Normalization','probability');
        plot(edges(1:end-1),N,'.-','color',map(jj,:))
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Displacement^2')
    ylabel('PDF')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(dis_interval));
    for jj = 1:length(dis_interval)
        all_disp = displacement_scale{length(tau),jj};
        log_space_min = log10(min(all_disp));
        if isinf(log_space_min)
            temp = unique(all_disp);
            log_space_min = log10(temp(2));
        end
        [N,edges] = histcounts(all_disp,logspace(log_space_min,log10(max(all_disp)),51),'Normalization','probability');
        plot(edges(1:end-1),N,'o-','color',map(jj,:))
    end
    for jj = 1:length(dis_interval)
        legend_string{jj} = ['\tau = ',num2str(dis_interval(jj))];
    end
    legend(legend_string)
    set(gca,'xscale','log','yscale','log')
    xlabel('Displacement')
    ylabel('PDF')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    saveas(gcf,fullfile(sub_loss_w_dir(1).folder,'displacement_distribution.fig'))

       
    % plot
    figure_width = 24;
    total_row = 3;
    total_column = 3;
    
    % [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
    EMH = 0.2;
    EMV = 0.4;
    [ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis( total_row, total_column, figure_width, EMH, 0.4, EMV, 0.4, 0.68, 0.5,5,4);
    % uniform FontSize and linewidth
    figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
        'color','w', 'position', [0, 0, figure_width, figure_hight], ...
        'PaperSize', [figure_width, figure_hight]); % this is the trick!
    
    subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('MSD')
    hold on
    map = jet(length(tau));
    for k=1:length(tau)
        loglog(tau{k},MSD{k},'color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
    legend(legend_string)
    xlabel('\tau')
    ylabel('{\Delta}r^2(\tau)')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    
    subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('Loss changes')
    %     yyaxis left
    hold on
    for k=1:length(tau)
        plot(1:length(delta_train_loss{k}),abs(delta_train_loss{k}),'color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    xlabel('Time (step)')
    ylabel('{\Delta}Train Loss')
    %legend({'train loss','test loss'})
    %     yyaxis right
    %     plot(1:length(delta_test_loss),abs(delta_test_loss))
    %     ylabel('{\Delta}Test Loss')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('{\Delta}Loss distribution')
    hold on
    
    for k=1:length(tau)
        [N,edges] = histcounts(delta_train_loss{k});
        plot(edges(1:end-1),N,'o-','color',map(k,:))
        % [N,edges] = histcounts(delta_test_loss{k});
        % plot(edges(1:end-1),N,'.-','color',map(k,:))
        legend_string{k} = ['train step:',num2str(1 + (k-1)*steps_in_part)];
        %legend_string{2*k} = ['test step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    clear legend_string
    %     set(gca,'xscale','log','yscale','log')
    xlabel('{\Delta}Loss')
    ylabel('Counts')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    
    Lecun = load(fullfile(sub_loss_w_dir(1).folder,'MSD_lecun.mat'));
    
    hold on
    legend_txt = [];      
    map = jet(length(Lecun.MSD_lecun));
    for jj = 1:length(Lecun.MSD_lecun)
        plot(Lecun.MSD_lecun{jj},'color',map(jj,:))
        legend_txt{end+1} = ['part=',num2str(jj)];        
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('\tau')
    ylabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    legend(legend_txt)
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')    
    
    subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);    
    loglog(Lecun.f_loss,Lecun.P1_loss,'K')
    xlabel('Frequency (step^{-1})')
    ylabel('Loss spectrum')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(contour_length));
    for k=1:length(tau)
        loglog(contour_length{k},MSD_forcontour{k},'color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    xlabel('Contour length')
    ylabel('{\Delta}r^2(\tau)')
    %     hold on
    %     P = polyfit(log(contour_length(contour_length<894)),log(MSD_forcontour(contour_length<894)),1);
    %     x_fit = logspace(log(min(contour_length)),log(894),20);
    %     y_fit = P(1)*x_fit + P(2);
    %     plot(x_fit,y_fit,'r:')
    %     title(['Slop:',num2str(P(1))])
    %     axis([min(contour_length),max(contour_length),min(MSD_forcontour),max(MSD_forcontour)])
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(tau));
    for k=1:length(tau)
        loglog(tau{k},Mean_contourlength{k},'color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    xlabel('\tau')
    ylabel('Contour length')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    
    subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(tau));
    for k=1:length(tau)
        % coarse grain for visualization
        [distance_coarse,MSL_coarse] = coarse_grain(distance{k},MSL_distance{k},50);
        plot(distance_coarse,MSL_coarse,'.-','color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    xlabel('Distance')
    ylabel('Mean squared loss')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    subaxis(total_row,total_column,3,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    map = jet(length(tau));
    for k=1:length(tau)
        % coarse grain for visualization
        [distance_coarse,MSL_coarse] = coarse_grain(contour_length{k},MSL_contourlength{k},50);
        plot(distance_coarse,MSL_coarse,'.-','color',map(k,:))
        legend_string{k} = ['step:',num2str(1 + (k-1)*steps_in_part)];
    end
%     legend(legend_string)
    xlabel('Contour length')
    ylabel('Mean squared loss')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out','xscale','log','yscale','log')
    
    
    saveas(gcf,fullfile(sub_loss_w_dir(1).folder,[d(ii).name(1:end-24),'_plot_',num2str(part),'.fig']))
    %saveas(gcf,fullfile(sub_loss_w_dir(1).folder,[d(ii).name,'_plot.svg']))
end

end


function MSD_feed = organize_data(all_sub_weights, all_sub_loss, all_test_sub_loss, d, GN_dir, part_num)
fprintf('Calculating part: %d',part_num)
% organize data to feed MSD calculation
empty_flag = cellfun(@isempty,all_sub_weights);
if sum(empty_flag) > 0 && sum(empty_flag) < 2
    all_sub_weights = all_sub_weights(~empty_flag);
    all_sub_loss = all_sub_loss(~empty_flag);
    all_test_sub_loss = all_test_sub_loss(~empty_flag);
elseif sum(empty_flag) > 1
    warning('Data %d has empty component and it is not the first one',part_num);
    MSD_feed = nan;
    save(fullfile(GN_dir(1).folder,[d.name(1:end-24),'_data_part_',num2str(part_num),'.mat']),'MSD_feed','-v7.3')
    return
end
tiny_step_num = size(all_sub_loss{1},2);
t = 1:(tiny_step_num*length(all_sub_loss));
MSD_feed = [cell2mat(all_sub_weights'),t',reshape(cell2mat(all_sub_loss),[],1)];
save(fullfile(GN_dir(1).folder,[d.name(1:end-24),'_data_part_',num2str(part_num),'.mat']),'MSD_feed','-v7.3')
end

function calculate_MSD_trajectory(MSD_feed, all_sub_loss, all_test_sub_loss, d, GN_dir, part_num)
fprintf('Calculating part: %d',part_num)
% pca trajectory
plot_trajectory_in_one_epoch(MSD_feed(:,1:end-2),fullfile(d.folder,d.name),part_num)

% calculate MSD
[MSD,Mean_contourlength,tau,MSD_forcontour,MSL_contourlength,contour_length,MSL_distance,distance,Displacement_all,Contour_length_all] = get_contour_lenth_MSD_loss(MSD_feed);

clear MSD_feed
MSD = gather(MSD);
Mean_contourlength = gather(Mean_contourlength);
tau = gather(tau);
MSD_forcontour = gather(MSD_forcontour);
MSL_contourlength = gather(MSL_contourlength);
contour_length = gather(contour_length);
MSL_distance = gather(MSL_distance);
distance = gather(distance);
Displacement_all = gather(Displacement_all);
Contour_length_all = gather(Contour_length_all);
disp('MSD and contour: done!')

% loss dynamics
loss = cell2mat(all_sub_loss);
loss = reshape(loss',[],1);
test_loss = cell2mat(all_test_sub_loss);
test_loss = reshape(test_loss',[],1);
% get the step difference of training loss
delta_train_loss = diff(loss);
% get the step difference of test loss
delta_test_loss = diff(test_loss);
% FFT
[f_loss,P1_loss,~] = get_fft_new(loss,1,'original');
[f_test_loss,P1_test_loss,~] = get_fft_new(test_loss,1,'original');

%save
save(fullfile(GN_dir(1).folder,[d.name(1:end-24),'_data_part_',num2str(part_num),'.mat']),'loss','test_loss','delta_train_loss','delta_test_loss',...
    'MSD','Mean_contourlength','tau','MSD_forcontour','MSL_contourlength','contour_length','MSL_distance',...
    'distance','Displacement_all','Contour_length_all','f_loss','P1_loss','f_test_loss','P1_test_loss','-append')
end

function [distance_coarse,MSL_coarse] = coarse_grain(distance,MSL,coarse_num)
% coarse grain for visualization
Dis_coarse = linspace(min(distance),max(distance),coarse_num+1);
for interval = 1:coarse_num
    MSL_coarse(interval) = mean(MSL((distance >= Dis_coarse(interval)) & (distance < Dis_coarse(interval+1))));
end
distance_coarse = movmean(Dis_coarse,2);
distance_coarse = distance_coarse(2:end);
end