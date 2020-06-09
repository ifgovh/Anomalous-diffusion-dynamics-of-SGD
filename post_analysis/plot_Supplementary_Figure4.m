% plot figures of Proj3
clear
close all
%% load data
d = dir('Z:\PRJ-THAP\epoch\resnet14_sgd_lr=*_bs=1024_wd=0_mom=0_save_epoch=*'); % 0.001,0.01,0.05,0.5
for ii = 1:length(d)
datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));

for part = 1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'MSD','tau');       
        
        MSD{ii}{part} = L.MSD;        
        tau{ii}{part} = L.tau;        
    catch        
        MSD{ii}{part} = nan;
        tau{ii}{part} = nan;        
    end
end
end

d = dir('X:\Project3\resnet14_sgd_lr=*_bs=1024_wd=0_mom=0_save_epoch=1');
ii=1;
datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));

for part = 1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'MSD','tau');       
        
        MSD{5}{part} = L.MSD;        
        tau{5}{part} = L.tau;        
    catch        
        MSD{5}{part} = nan;
        tau{5}{part} = nan;        
    end
end

%% load data
% all data are from /Project3/MSD_exponent_DNN.xlsx
keySet = {'0.001','0.01','0.05','0.1','0.5'};    
valueSet_exp1 = [2,1.833333333,1.7,1.387755102,1.4];
valueSet_exp2 = [2,1.7,0.676666667,0.69,0.726666667];
M_exp1 = containers.Map(keySet,valueSet_exp1);
M_exp2 = containers.Map(keySet,valueSet_exp2);
%% plot learning-rate-affected superdiffusion
figure_width = 16;
total_row = 3;
total_column = 3;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 1.5;
EMH = 0.15;
EMV = 0.28;
MLR = 0.72;
MBR = 0.8;
[ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis( total_row, total_column, figure_width, EMH, 0.4, EMV, 0.4, MLR, MBR);

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!
% lr vs 1st MSD exponent
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
brewermap('Oranges');
map = brewermap(5);
plot([0.001,0.01,0.05,0.1,0.5],[M_exp1('0.001'),M_exp1('0.01'),M_exp1('0.05'),M_exp1('0.1'),M_exp1('0.5')],'marker','o','linestyle','-','color',map(end,:),'linewidth',linewidth)
y = ylabel('MSD exponent 1');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
% set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

% lr vs 2st MSD exponent
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
brewermap('Oranges');
map = brewermap(5);
plot([0.001,0.01,0.05,0.1,0.5],[M_exp2('0.001'),M_exp2('0.01'),M_exp2('0.05'),M_exp2('0.1'),M_exp2('0.5')],'marker','o','linestyle','-','color',map(end,:),'linewidth',linewidth)
y = ylabel('MSD exponent 2');
x = xlabel('Learning rate (\eta)');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

% loss vs time for different lrs
load('Z:\PRJ-THAP\epoch\resnet14_sgd_lr_loss.mat')
subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
map = [130,57,53;137,190,178;201,186,131;222,211,140;222,156,83]./256;
hold on
for ii = 1:length(lr)
    plot(loss{ii},'color',map(ii,:));
end
legend({'\eta=0.001','\eta=0.01','\eta=0.05','\eta=0.1','\eta=0.5'})
legend box off
y = ylabel('Loss');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
x = xlabel('Time (step)');
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')


%% MSD-lr 0.001
cc = subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num%length(tau)
    loglog(tau{1}{k},MSD{1}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,7e-8,1])
% xlabel('\tau')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'ytick',[1e-7,1e-4,1])

% a single MSD curve
insect = axes('position',[0.40793388429752 0.871926843317972 0.104462809917356 0.101636904761905]);
axis(insect);box on;
hold on
for k = [1,round(select_num/3),select_num]
    loglog(tau{1}{k},MSD{1}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,7e-8,1])
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

%% MSD-lr 0.01
cc = subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num%length(tau)
    loglog(tau{2}{k},MSD{2}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 1.4e-5 1e3])
% xlabel('\tau')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
text(-0.18,1.15,'e','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'ytick',[1e-4,1,1e3])

% a single MSD curve
insect = axes('position',[0.40793388429752 0.540219526244801 0.104462809917356 0.101636904761905]);
axis(insect);box on;
hold on
for k = [1,round(select_num/3),select_num]
    loglog(tau{2}{k},MSD{2}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 1.4e-5 1e3])
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

%% MSD-lr 0.05
cc = subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num%length(tau)
    loglog(tau{3}{k},MSD{3}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 2e-3 2e2])
x = xlabel('\tau');
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]); 
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'f','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'ytick',[1e-2,1,1e2])

% a single MSD curve
insect = axes('position',[0.407933884297521 0.208491126302779 0.104462809917356 0.101636904761905]);
axis(insect);box on;
hold on
for k = [1,round(select_num/3),select_num]
    loglog(tau{3}{k},MSD{3}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 2e-3 2e2])
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

%% MSD-lr 0.1
cc = subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num%length(tau)
    loglog(tau{5}{k},MSD{5}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 1e-2 1e2])
% xlabel('\tau')
% ylabel('{\Delta}r^2(\tau)')
text(-0.18,1.15,'g','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'ytick',[1e-2,1,1e2])

% a single MSD curve
insect = axes('position',[0.745123966942148 0.871921572600759 0.104462809917356 0.101636904761905]);
axis(insect);box on;
hold on
for k = [1,round(select_num/3),select_num]
    loglog(tau{5}{k},MSD{5}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1 1e3 1e-2 1e2])
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

%% MSD-lr 0.5
cc = subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=1:select_num%length(tau)
    loglog(tau{4}{k},MSD{4}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,0.8e-1,200])
x = xlabel('\tau');
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
% ylabel('{\Delta}r^2(\tau)')
text(-0.18,1.15,'h','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'ytick',[1e-1,1,10,1e2])
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e3])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1)  originalSize(2)-0.08  originalSize(3) 0.015],'location','south');
T = title(c,'t_w (step)','fontsize',fontsize);
set(T,'Units', 'Normalized', 'Position', [0.5,-4.5, 0]);
set(gca, 'Position', originalSize);

% a single MSD curve
insect = axes('position',[0.745123966942148 0.540219526244801 0.104462809917356 0.101636904761905]);
axis(insect);box on;
hold on
for k = [1,round(select_num/3),select_num]
    loglog(tau{4}{k},MSD{4}{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,0.8e-1,200])
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

set(gcf, 'PaperPositionMode', 'auto');

% output
print( '-painters' ,'X:\Project3\outputfigures\lr_diffusion.eps','-depsc','-r300')
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/lr_diffusion.svg','-dsvg','-r300')

