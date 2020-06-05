function SGD_analysis_step_level_lecun(varargin)
d = dir('resnet14_512');
% Loop number for PBS array job
loop_num = 0;

for ii = 1:length(d)
    % For PBS array job
    loop_num = loop_num + 1;
    if nargin ~= 0
        PBS_ARRAYID = varargin{1};
        if loop_num ~=  PBS_ARRAYID
            continue;
        end
    end
    
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*.mat'));
    
    % TRAJECTORY
    Trajectory = [];
    loss = [];
    k = 1;
    for part = 1:length(sub_loss_w_dir)
        load(fullfile(sub_loss_w_dir(ii).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'MSD_feed')
        Trajectory = [Trajectory;MSD_feed];
        loss = [loss;MSD_feed(:,end)];
        % if the variable is too large, split it.
        attrib = whos('Trajectory');
        if attrib.bytes/1024^3 > 5
            [MSD_lecun{k},contour_length_lecun{k},dL_lecun{k}] = get_contour_lenth_MSD_loss_lecun_version(Trajectory);
            Trajectory = [];
            k = k + 1;
        end
    end
    % get the step difference of training loss
    delta_train_loss = diff(loss);
   
    % FFT
    [f_loss,P1_loss,~] = get_fft_new(loss,1,'original');
    
    save(fullfile(sub_loss_w_dir(1).folder,'MSD_lecun.mat'),'MSD_lecun',...
        'contour_length_lecun','dL_lecun','delta_train_loss','f_loss','P1_loss')
    
    figure_width = 32;
    total_row = 3;
    total_column = 2;
    
    % [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
    EMH = 0.2;
    EMV = 0.4;
    
    [ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis( total_row, total_column, figure_width, EMH, 0.4, EMV, 0.4, 0.68, 0.5,5,4);
    % uniform FontSize and linewidth
    figure('NumberTitle','off','name', 'displacement_distri', 'units', 'centimeters', ...
        'color','w', 'position', [0, 0, figure_width, figure_hight], ...
        'PaperSize', [figure_width, figure_hight]); % this is the trick!
    
    subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    map = jet(length(MSD_lecun));
    for jj = 1:length(MSD_lecun)
        plot(MSD_lecun{jj},'color',map(jj,:))
        legend_txt{end+1} = ['t_w=',num2str(jj)];
        
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('\tau')
    ylabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    legend(legend_txt)
    
    subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    for jj = 1:length(MSD_lecun)
        try
            plot(contour_length_lecun{jj},MSD_lecun{jj},'color',map(jj,:))
            legend_txt{end+1} = ['t_w=',num2str(jj)];
        catch
        end
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Contour length')
    ylabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    legend(legend_txt)
    
    subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    for jj = 1:length(MSD_lecun)
        try
            plot(contour_length_lecun{jj},dL_lecun{jj},'color',map(jj,:))
            legend_txt{end+1} = ['t_w=',num2str(jj)];
        catch
        end
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('Contour length')
    ylabel('$\Delta Loss$','interpreter','latex')
    legend(legend_txt)
    
    subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    hold on
    legend_txt = [];
    for jj = 1:length(MSD_lecun)
        try
            plot(MSD_lecun{jj},dL_lecun{jj},'color',map(jj,:))
            legend_txt{end+1} = ['t_w=',num2str(jj)];
        catch
        end
    end
    set(gca,'xscale','log','yscale','log')
    xlabel('$\Delta(t_w,t_w+\tau)$','interpreter','latex')
    ylabel('$\Delta Loss$','interpreter','latex')
    legend(legend_txt)   
    
    subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    plot(loss,'k-')
    set(gca,'xscale','log')
    xlabel('time (steps)')
    ylabel('Loss')
    
    
    subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
        'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
    plot(f_loss,P1_loss,'k-')
    set(gca,'xscale','log','yscale','log')
    xlabel('Frequency (steps^{-1})')
    ylabel('Power of loss')
    
    
    saveas(gcf,fullfile(sub_loss_w_dir(1).folder,'MSD_lecun.fig'))    
    
    
    % LOSS
end


