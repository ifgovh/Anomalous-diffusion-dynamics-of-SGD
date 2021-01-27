clear
close all
%% load data
d = dir('/project/RDS-FSC-cortical-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet*');
ii = 1;
datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
select_num = 24;
for part = [1,round(select_num/5),select_num]%1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),...
            'F_grads','F_full_batch_grad','F_gradient_noise_norm','Grads','Full_grads','G_noise_norm');
        F_grads{part} = L.F_grads;
        F_full_batch_grad{part} = L.F_full_batch_grad;
        F_gradient_noise_norm{part} = L.F_gradient_noise_norm;
        Grads{part} = L.Grads;
        Full_grads{part} = L.Full_grads;
        G_noise_norm{part} = L.G_noise_norm;
    catch
        F_grads{part} = nan;
        F_full_batch_grad{part} = nan;
        F_gradient_noise_norm{part} = nan;
        Grads{part} = nan;
        Full_grads{part} = nan;
        G_noise_norm{part} = nan;
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
EMH = 0.1;
EMV = 0.4;
MLR = 0.65;
MBR = 0.8;

[ figure_hight, SV, SH, MT,MB,ML,MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.6,MLR,MBR);
figure('NumberTitle','off','name', 'Reproduction', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!

% gradient
G_dir = dir(fullfile(d(ii).folder,'all_gradients.mat'));
load(fullfile(G_dir.folder,G_dir.name))
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
factor = 0.2576/782.7;% shift the wrong normalization of matlab
[N,edges] = histcounts(grad,651,'normalization','pdf');
plot(edges(2:end),N*factor,'ko')
plot(linspace(-0.05,0.05,5000),pdf(F,linspace(-0.05,0.05,5000))*factor,'m-');
ttt = grad(randperm(numel(grad),2000));
F_G = fitdist(ttt(:),'normal');
plot(linspace(-0.05,0.05,5000),pdf(F_G,linspace(-0.05,0.05,5000))*factor,'b-');
xlim([-0.006,0.006])
y = ylabel('PDF');
x = xlabel('Gradient');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
legend({'Data','S. dist.','G. dist.'})
legend boxoff
box off
text(-0.18,1.15,'(a)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])

% log-log scale
insect = axes('position',[0.22191928075833 0.337979094076655 0.080428938438048 0.108139095130109]);
axis(insect);box on;
plot(linspace(0,0.2,1000),pdf(F,linspace(0,0.2,1000)),'m-');
% axis([1,1e3,1e-2,1e2])
set(get(gca,'Ylabel'),'rotation',-45)
set(gca,'xscale','log','yscale','linear','tickdir','out','TickLength',[TickLength 0.035])


% \Delta L vs t
load(fullfile(d(1).folder,'trans_ex_avalan.mat'))
delta_loss_temp = abs(diff(loss_all{2}));
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
kkk = 1;
map = [0,90,171;71,31,31;255,66,93]./255;
for k = [1,round(select_num/5),select_num]
    plot(-kkk*0.2+delta_loss_temp(((k-1)*1000+100):k*1000-500),'color',map(kkk,:),'linewidth',linewidth)
    kkk = kkk + 1;
end
x = xlabel('Relative time (step)');
y = ylabel('|{\Delta}L|');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
text(-0.18,1.15,'(b)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])


% movvar vs t
subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
var_win = 100;
instant_var = movvar(loss_all{2},[0,var_win]);
instant_var =  instant_var(1:var_win:end);
plot(1:var_win:numel(loss_all{2}),instant_var,'linewidth',linewidth,'color','k')
axis([-200,24000,0,0.005])
x = xlabel('Time (step)');
y = ylabel('Moving variance');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
text(-0.18,1.15,'(c)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])

% gradient
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on

map = [0,90,171;71,31,31;255,66,93]./255;
kkk = 1;
for k = [1,round(select_num/5),select_num]
    [N,edges] = histcounts(Grads{k},'normalization','pdf');
    data_points{kkk} = plot(edges(2:10:end),N(1:10:end),'color',map(kkk,:),'marker','o','linestyle','none');
    plot(linspace(-0.2,0.2,5000),pdf(F_grads{k},linspace(-0.2,0.2,5000)),'color',map(kkk,:));
    kkk = kkk + 1;
end
xlim([-0.05,0.05])
legend([data_points{1},data_points{2},data_points{3}], {'t_w=1 step','t_w=5001','t_w=24000'})
legend boxoff
y = ylabel('PDF');
x = xlabel('Minibatch Gradient');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
text(-0.18,1.15,'(d)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035],'yscale','linear')

subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on

map = [0,90,171;71,31,31;255,66,93]./255;
kkk = 1;
for k = [1,round(select_num/5),select_num]
    [N,edges] = histcounts(Full_grads{k},'normalization','pdf');
    data_points{kkk} = plot(edges(2:10:end),N(1:10:end),'color',map(kkk,:),'marker','o','linestyle','none');
    plot(linspace(-0.05,0.05,5000),pdf(F_full_batch_grad{k},linspace(-0.05,0.05,5000)),'color',map(kkk,:));
    kkk = kkk + 1;
end
legend([data_points{1},data_points{2},data_points{3}],{'t_w=1 step','t_w=5001','t_w=24000'})
legend boxoff
y = ylabel('PDF');
x = xlabel('Estimated full-batch Gradient');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
text(-0.18,1.15,'(e)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035],'yscale','linear')

subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on

map = [0,90,171;71,31,31;255,66,93]./255;
kkk = 1;
for k = [1,round(select_num/5),select_num]
    [N,edges] = histcounts(G_noise_norm{k},'normalization','pdf');
    data_points{kkk} = plot(edges(2:10:end),N(1:10:end),'color',map(kkk,:),'marker','o','linestyle','none');
    plot(linspace(-0.05,0.05,5000),pdf(F_gradient_noise_norm{k},linspace(-0.05,0.05,5000)),'color',map(kkk,:));
    kkk = kkk + 1;
end
legend([data_points{1},data_points{2},data_points{3}],{'t_w=1 step','t_w=5001','t_w=24000'})
legend boxoff
y = ylabel('PDF');
x = xlabel('Minibatch Gradient');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, xlable_shift, 0]);
text(-0.18,1.15,'(f)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','TickLength',[TickLength 0.035])

set(gcf, 'PaperPositionMode', 'auto');

% output
% print('-painters' ,'fig5.svg','-dsvg','-r300')