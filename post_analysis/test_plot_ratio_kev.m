clear
close all
%% load data
% initialize saving data
resnet20_epoch500.nn_type = 'resnet20_epoch500';
% d = dir('/project/RDS-FSC-THAP-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet14*');
d = dir(strcat('/project/RDS-FSC-THAP-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/',resnet20_epoch500.nn_type,'*'));
alpha_ls = zeros(1,4); % same as the length of ii
%for ii = 1:length(d)
% resnet14_epoch list with MSD, etc
MSD_ls = [1];
for iii = 1:length(MSD_ls)
    ii = MSD_ls(iii);
    clearvars -except d ii iii alpha_ls MSD_ls resnet20_epoch500
    close all

datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
select_num = length(datax_dir)-1;
for part = [1,round(select_num/5),select_num]%1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),...
            'MSD','tau','MSD_noaverage','tau_noaverage','F_grads','F_full_batch_grad','F_gradient_noise_norm','Grads','Full_grads','G_noise_norm');
        
        MSD{part} = L.MSD;
        tau{part} = L.tau;
       
        MSD_noaverage{part} = L.MSD_noaverage;
        tau_noaverage{part} = L.tau_noaverage;
        
        F_grads{part} = L.F_grads;
        F_full_batch_grad{part} = L.F_full_batch_grad;
        F_gradient_noise_norm{part} = L.F_gradient_noise_norm;
        Grads{part} = L.Grads;
        Full_grads{part} = L.Full_grads;
        G_noise_norm{part} = L.G_noise_norm;
    catch        
        MSD{part} = nan;
        tau{part} = nan;
        
        MSD_noaverage{part} = nan;
        tau_noaverage{part} = nan;
        
        F_grads{part} = nan;
        F_full_batch_grad{part} = nan;
        F_gradient_noise_norm{part} = nan;
        Grads{part} = nan;
        Full_grads{part} = nan;
        G_noise_norm{part} = nan;
    end

%kkk=1;
% for k = [1,round(select_num/5),select_num]
%     loglog(tau{k},MSD{k},'color',map(kkk,:),'linewidth',linewidth)
%     kkk=kkk+1;
% end
end
MSD_regime1 = MSD{1};
MSD_regime1 = 1./double(MSD_regime1);
%tau_regime1 = tau{1};
%tau_regime1 = double(tau_regime1);
% this is because the x-axis must have a number greater than 1
% if min(tau_regime1) <= 1
%     tau_regime1 = tau_regime1 + 1;
% end
alpha = plfit(MSD_regime1,'nosmall');
% estimate first regime tail index 
alpha_ls(iii) = 1/(alpha-1);
end

%% save data first (do this manually), make sure to name the variable as the above
% lr/batch size ratio (done manually)
lb_ratio = [0.15/128,0.1/128,0.3/128,0.5/128];
resnet20_epoch500.ratio_ls = lb_ratio;
resnet20_epoch500.alpha_ls = alpha_ls;
save('ratio_alpha_data.mat','-struct','resnet20_epoch500');

%% plot superdiffusion
% figure_width = 20;
% total_row = 2;
% total_column = 3;
% % uniform FontSize and linewidth
% fontsize = 10;
% linewidth = 0.7;
% ylable_shift = -0.2;
% xlable_shift = -0.15;
% TickLength = 0.03;
% EMH = 0.1;
% EMV = 0.4;
% MLR = 0.65;
% MBR = 0.8;
% 
% [ figure_hight, SV, SH, MT,MB,ML,MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.6,MLR,MBR);
% figure('NumberTitle','off','name', 'Reproduction', 'units', 'centimeters', ...
%     'color','w', 'position', [0, 0, figure_width, figure_hight], ...
%     'PaperSize', [figure_width, figure_hight]); % this is the trick!
% 
% % eta/batch_size ratio plot
% subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% hold on
% select_num = length(MSD);
% map = [0,90,171;71,31,31;255,66,93]./255;
% 
% xlabel('\eta/B')
% y = ylabel('\alpha in the first regime');
% set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
% text(-0.18,1.15,'(a)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])
% 
% plot(lb_ratio,alpha_ls,'linewidth',linewidth,'o-');

% axis([1,1e3,1e-2,1e2])
%legend({'t_w=1 step','t_w=5001','t_w=24000'})
%legend boxoff

% %no temporal average MSD
% subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% hold on
% select_num = length(MSD);
% map = [0,90,171;71,31,31;255,66,93]./255;
% kkk=1;
% for k = [1,round(select_num/5),select_num]
%     loglog(tau_noaverage{k},MSD_noaverage{k},'color',map(kkk,:),'linewidth',linewidth)
%     kkk=kkk+1;
% end
% % axis([1,1e3,1e-2,1e2])
% legend({'t_w=1 step','t_w=5001','t_w=24000'})
% legend boxoff
% xlabel('\tau no average (step)')
% y = ylabel('{\Delta}r^2(\tau)');
% set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
% text(-0.18,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])

% % gradient 'F_grads','F_full_batch_grad','F_gradient_noise_norm','Grads','Full_grads','G_noise_norm'
% subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% hold on
% select_num = length(MSD);
% map = [0,90,171;71,31,31;255,66,93]./255;
% kkk = 1;
% for k = [1,round(select_num/5),select_num]
% [N,edges] = histcounts(Grads{k},651,'normalization','pdf');
% plot(edges(2:end),N,'color',map(kkk,:),'marker','o')
% plot(linspace(-0.05,0.05,5000),pdf(F_grads{k},linspace(-0.05,0.05,5000)),'color',map(kkk,:));
% kkk = kkk + 1;
% end
% legend({'t_w=1 step','t_w=5001','t_w=24000'})
% legend boxoff
% y = ylabel('PDF');
% x = xlabel('Minibatch Gradient');
% set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
% set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
% text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])
% 
% subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% hold on
% select_num = length(MSD);
% map = [0,90,171;71,31,31;255,66,93]./255;
% kkk = 1;
% for k = [1,round(select_num/5),select_num]
% [N,edges] = histcounts(Full_grads{k},651,'normalization','pdf');
% plot(edges(2:end),N,'color',map(kkk,:),'marker','o')
% plot(linspace(-0.05,0.05,5000),pdf(F_full_batch_grad{k},linspace(-0.05,0.05,5000)),'color',map(kkk,:));
% kkk = kkk + 1;
% end
% legend({'t_w=1 step','t_w=5001','t_w=24000'})
% legend boxoff
% y = ylabel('PDF');
% x = xlabel('Estimated full-batch Gradient');
% set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
% set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
% text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])
% 
% subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% hold on
% select_num = length(MSD);
% map = [0,90,171;71,31,31;255,66,93]./255;
% kkk = 1;
% for k = [1,round(select_num/5),select_num]
% [N,edges] = histcounts(G_noise_norm{k},651,'normalization','pdf');
% plot(edges(2:end),N,'color',map(kkk,:),'marker','o')
% plot(linspace(-0.05,0.05,5000),pdf(F_gradient_noise_norm{k},linspace(-0.05,0.05,5000)),'color',map(kkk,:));
% kkk = kkk + 1;
% end
% legend({'t_w=1 step','t_w=5001','t_w=24000'})
% legend boxoff
% y = ylabel('PDF');
% x = xlabel('Minibatch Gradient');
% set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
% set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
% text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])

% set(gcf, 'PaperPositionMode', 'auto');
% 
% % output
% %saveas(gcf,sprintf('test_noavearge_MSD_gradient_%d.fig',ii))
% saveas(gcf,sprintf('lr_batch_ratio_resnet14.fig'))