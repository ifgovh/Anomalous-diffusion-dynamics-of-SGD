clear
close all
%% load data
d = dir('/project/RDS-FSC-cortical-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet14_epoch5000_*');
ii = 1;

datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));

for part = 1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),...
            'MSD','tau','MSD_noaverage','tau_noaverage');        
        
        MSD{part} = L.MSD;        
        tau{part} = L.tau;
        
        MSD_noaverage{part} = L.MSD_noaverage;
        tau_noaverage{part} = L.tau_noaverage;
        
    catch
        
        MSD{part} = nan;
        tau{part} = nan;
        
        MSD_noaverage{part} = nan;
        tau_noaverage{part} = nan;
    end
end
%% plot superdiffusion
figure_width = 20;
total_row = 2;
total_column = 3;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 0.7;
ylable_shift = -0.2;
xlable_shift = -0.15;
TickLength = 0.03;
EMH = 0.4;
EMV = 0.4;
MLR = 0.65;
MBR = 0.8;

[ figure_hight, SV, SH, MT,MB,ML,MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.6,MLR,MBR);
figure('NumberTitle','off','name', 'Reproduction', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!

%three curves of MSD in 500 epoch trial
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k = [1,round(select_num/5),select_num]
    loglog(tau{k},MSD{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,1e-2,1e2])
legend({'t_w=1 step','t_w=5001','t_w=23001'})
legend boxoff
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'(a)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])

%all curves of MSD
cc = subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
for k = 1:select_num
    loglog(tau{k},MSD{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,1e-2,1e2])
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'(b)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e3])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1) + originalSize(3) + 0.01  originalSize(2) 0.01 originalSize(4) ],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
set(gca, 'Position', originalSize);


% tau=1e4 MSD
cc = subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
d_long = dir('/project/RDS-FSC-THAP-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet14_512/_*mat');% one should change the parameter "steps_in_part" in SGD_analysis_step_level.m
hold on
select_num = 5;
map = jet(select_num);
for k=1:select_num%length(tau)
    long_tau = load(fullfile(d_long(k).folder,d_long(k).name),'MSD','tau');
    loglog(long_tau.tau,long_tau.MSD,'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e4,1e-2,6e2])
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.13,'(d)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000,1e4],'TickLength',[TickLength 0.035])
% legend({'t_w=1 step','t_w=10001 steps','t_w=20001 steps','t_w=30001 steps','t_w=40001 steps'})
% legend boxoff
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e4])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1) + originalSize(3) + 0.01  originalSize(2) 0.01 originalSize(4) ],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
set(gca, 'Position', originalSize);

% dynamical exponent
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num
    FY = (log(MSD{k}(1:end-20)) - log(MSD{k}(21:end)))./(log(tau{k}(1:end-20)) - log(tau{k}(21:end)));
    plot(tau{k}(1:end-20),FY,'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,0.5,2.5])
xlabel('\tau (step)')
ylabel('\beta(\tau)')
text(-0.18,1.1,'(c)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','linear','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])
% one dynamical exponent
insect = axes('position',[0.074 0.316552388432713 0.09 0.129036460946836]);
axis(insect);box on;
hold on
for k = [1,round(select_num/5),select_num]
    FY = (log(MSD{k}(1:end-20)) - log(MSD{k}(21:end)))./(log(tau{k}(1:end-20)) - log(tau{k}(21:end)));
    plot(tau{k}(1:end-20),FY,'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,0.5,2.5])
% x = xlabel('\tau');
% set(x, 'Units', 'Normalized', 'Position', [0.5, -0.1, 0]);
% y = ylabel('\beta(\tau)');
% set(y, 'Units', 'Normalized', 'Position', [-0.22, 0.5, 0]);
set(gca,'xscale','log','yscale','linear','xtick',[],'ytick',[])

% epoch 5000 MSD
subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = length(MSD);
map = [0,90,171;71,31,31;255,66,93]./255;
kkk=1;
for k = [1,round(select_num/10),select_num]
    loglog(tau{k},MSD{k},'color',map(kkk,:),'linewidth',linewidth)
    kkk=kkk+1;
end
% axis([1,1e3,1e-2,1e2])
legend({'t_w=1 step','t_w=23001','t_w=230001'})
legend boxoff
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'(e)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])

%no temporal average MSD
subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = [0,90,171;71,31,31;255,66,93]./255;
kkk=1;
for k = [1,round(select_num/5),select_num]
    loglog(tau_noaverage{k},MSD_noaverage{k},'color',map(kkk,:),'linewidth',linewidth)
    kkk=kkk+1;
end
% axis([1,1e3,1e-2,1e2])
legend({'t_w=1 step','t_w=5001','t_w=23001'})
legend boxoff
xlabel('\tau (step)')
y = ylabel('No time-average {\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'(f)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])

set(gcf, 'PaperPositionMode', 'auto');

% output
%print('-painters' ,'fig2.svg','-dsvg','-r300')