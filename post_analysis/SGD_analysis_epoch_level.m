% calculate the statistical property of SGD walk
function SGD_analysis_epoch_level(varargin)
% read in data
d = dir('*net*');
calculate = 1;

% Loop number for PBS array job
loop_num = 0;
    
for ii = 1:length(d)
    GN_dir = dir(fullfile(d(ii).folder,d(ii).name,'*gradient_noise*mat'));
    output_dir = dir(fullfile(d(ii).folder,d(ii).name,'*datax*mat'));
    % For PBS array job
    loop_num = loop_num + 1;
    if nargin ~= 0
        PBS_ARRAYID = varargin{1};
        if loop_num ~=  PBS_ARRAYID
            continue;
        end
    end
    
    if isempty(GN_dir) || ~isempty(output_dir)
        continue
    end
    if calculate
        load(fullfile(GN_dir.folder,GN_dir.name))
        % get the step difference of training loss
        delta_train_loss = diff(training_history(:,1));
        % get the step difference of test loss
        delta_test_loss = diff(testing_history(:,1));
        % organise the gradient noise
        GN = cell2mat(train_noise_norm(:,1));
        % organise the tail index
        alpha = cell2mat(train_noise_norm(:,2));
        
        weight_dir = dir(fullfile(d(ii).folder,d(ii).name,'*all_weights.mat'));
        load(fullfile(weight_dir.folder,weight_dir.name))
        % organise the weights
        if size(weight,1) < 600
            all_weights_tmp = cellfun(@(x) reshape(x,1,[]) ,weight,'UniformOutput',false);
        else
            all_weights_tmp = cellfun(@(x) reshape(x,1,[]) ,weight(1:1400,:),'UniformOutput',false);
        end
        all_weights = cell2mat(all_weights_tmp);
        clear all_weights_tmp
        disp('Organise weights: done!')
        % contour length related and MSD
        t = 1:size(all_weights,1);
        [MSD,tau,MSD_forcontour,contour_length,Displacement_all,Contour_length_all] = get_contour_lenth_MSD([all_weights,t']);
        disp('MSD and contour: done!')
        
        %save
        save(fullfile(GN_dir.folder,[d(ii).name,'_datax.mat']),'delta_train_loss','delta_test_loss',...
            'GN','alpha','all_weights','MSD','tau','MSD_forcontour','contour_length','Displacement_all','Contour_length_all','-v7.3')
    else
        load(fullfile(GN_dir.folder,[d(ii).name,'_datax.mat']),'delta_train_loss','delta_test_loss',...
            'GN','alpha','MSD','tau','MSD_forcontour','contour_length','Displacement_all','Contour_length_all')
    end
    
    % starting moment of square displacement
    t_start = [2.^(0:8),400];
   
    % displacement distribution
    dis_interval = 2.^(0:8);
    for jj = 1:length(dis_interval)
        eval(['displacement_',num2str(dis_interval(jj)),' = diag(Displacement_all,',num2str(dis_interval(jj)),');']);
    end  
    
    % plot
    figure_width = 48;
    total_row = 3;
    total_column = 4;
    
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
    loglog(tau,MSD)
    xlabel('\tau')
    ylabel('{\Delta}r^2(\tau)')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('disp. distri.')
    hold on
    legend_txt = [];
    for jj = 1:length(dis_interval)
        eval(['[N,edges] = histcounts(displacement_',num2str(dis_interval(jj)),');']);
        plot(edges(1:end-1),N)
        legend_txt{end+1} = ['\tau=',num2str(dis_interval(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Displacement')
    ylabel('Counts')
    legend(legend_txt)
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('Loss changes')
    yyaxis left
    plot(1:length(delta_train_loss),abs(delta_train_loss))    
    xlabel('Time (step)')
    ylabel('{\Delta}Train Loss')
    %legend({'train loss','test loss'})
    yyaxis right    
    plot(1:length(delta_test_loss),abs(delta_test_loss))
    ylabel('{\Delta}Test Loss')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,4,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('D^2 distribution')
    
    hold on
    legend_txt = [];
    for jj = 1:length(dis_interval)
        eval(['[N,edges] = histcounts(displacement_',num2str(dis_interval(jj)),'.^2);']);
        plot(edges(1:end-1),N)
        legend_txt{end+1} = ['\tau=',num2str(dis_interval(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Displacement^2')
    ylabel('Counts')
    legend(legend_txt) 
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);    
    loglog(contour_length,MSD_forcontour)
    xlabel('Contour length')
    ylabel('{\Delta}r^2(\tau)')
    hold on
    P = polyfit(log(contour_length(contour_length<894)),log(MSD_forcontour(contour_length<894)),1);
    x_fit = logspace(log(min(contour_length)),log(894),20);
    y_fit = P(1)*x_fit + P(2);
    plot(x_fit,y_fit,'r:')
    title(['Slop:',num2str(P(1))])
    axis([min(contour_length),max(contour_length),min(MSD_forcontour),max(MSD_forcontour)])
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    title('{\Delta}Loss distribution')
    [N,edges] = histcounts(delta_train_loss);
    plot(edges(1:end-1),N,'ko-')
    hold on
    [N,edges] = histcounts(delta_test_loss);
    plot(edges(1:end-1),N,'ro-')
    %     set(gca,'xscale','log','yscale','log')
    xlabel('{\Delta}Loss')
    ylabel('Counts')
    legend({'train loss','test loss'})
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    plot(1:length(alpha),alpha,'.-')
    ylabel('Tail index')
    xlabel('Time (step)')
    set(gca,'linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,4,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    [N,edges] = histcounts(GN(:),logspace(log(min(GN(:))),log(max(GN(:))),51));
    plot(edges(1:end-1),N,'ro-')
    ylabel('Counts')
    xlabel('Gradient noise norm')
    set(gca,'xscale','log','yscale','log','linewidth',1,'fontsize',12,'tickdir','out')
    
    subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    for jj = 1:length(t_start)
        plot(Displacement_all(t_start(jj),(1 + t_start(jj)):end));        
        legend_txt{end+1} = ['t_w=',num2str(t_start(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('\tau')
    ylabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    legend(legend_txt) 
    
    subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
     hold on
    legend_txt = [];
    for jj = 1:length(t_start)
        plot(Contour_length_all(t_start(jj),:),Displacement_all(t_start(jj),:));        
        legend_txt{end+1} = ['t_w=',num2str(t_start(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Contour length')
    ylabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    legend(legend_txt)
    
    subaxis(total_row,total_column,3,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    for jj = 1:length(t_start)
        plot(Displacement_all(t_start(jj),(1 + t_start(jj)):end)./(mean(GN(t_start(jj):500,:),2).^2)');
        legend_txt{end+1} = ['t_w=',num2str(t_start(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('\tau')
    ylabel('$\Delta(t_w,t_w+\tau)/D(t_w)$','interpreter','latex')
    legend(legend_txt)
    
    subaxis(total_row,total_column,4,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
     hold on
    legend_txt = [];
    for jj = 1:length(t_start)
        plot(Contour_length_all(t_start(jj),2:end),Displacement_all(t_start(jj),2:end)./(mean(GN(1:(size(Displacement_all,2)-1),:),2).^2)');        
        legend_txt{end+1} = ['t_w=',num2str(t_start(jj))];
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Contour length')
    ylabel('$\Delta(t_w,t_w+\tau)/D(t_w)$','interpreter','latex')
    legend(legend_txt)
    
    saveas(gcf,fullfile(GN_dir.folder,[d(ii).name,'_plot.fig']))
    saveas(gcf,fullfile(GN_dir.folder,[d(ii).name,'_plot.svg']))
end
end



