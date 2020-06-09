clear
close all
%% load data
d = dir('*net*');
ii = 2;

datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));

for part = 1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'loss','test_loss','delta_train_loss','delta_test_loss',...
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

% plot superdiffusion for supplementary figure 1
figure_width = 20;
total_row = 1;
total_column = 2;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 0.7;
ylable_shift = -0.2;
xlable_shift = -0.15;
TickLength = 0.03;
% [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
EMH = 0.3;
EMV = 0.4;
MLR = 0.6;
MBR = 0.4;

[ figure_hight, SV, SH, MT,MB,ML,MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.6,MLR,MBR);
figure('NumberTitle','off','name', 'Reproduction', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!

%MSD
cc = subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k = 1:select_num
    loglog(tau{k},MSD{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,1e-2,1e2])
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e3])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1) + originalSize(3) + 0.01  originalSize(2) 0.01 originalSize(4) ],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
set(gca, 'Position', originalSize);


% long MSD
cc = subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
d_long = dir('/import/headnode1/gche4213/Project3/other/resnet14_512/_*mat');
hold on
select_num = 5;
map = jet(select_num);
for k=1:select_num%length(tau)
    load(fullfile(d_long(k).folder,d_long(k).name),'MSD','tau')
    loglog(tau,MSD,'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e4,1e-2,6e2])
xlabel('\tau (step)')
y = ylabel('{\Delta}r^2(\tau)');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.13,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000,1e4])
% legend({'t_w=1 step','t_w=10001 steps','t_w=20001 steps','t_w=30001 steps','t_w=40001 steps'})
% legend boxoff
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e4])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1) + originalSize(3) + 0.01  originalSize(2) 0.01 originalSize(4) ],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
set(gca, 'Position', originalSize);

set(gcf, 'PaperPositionMode', 'auto');

% output
print('-painters' ,'superdiffusion_SI.svg','-dsvg','-r300')