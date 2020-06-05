% plot figures of Proj3
clear
close all
%% load data
d = dir('/import/headnode1/gche4213/Project3/*net*');
ii = 2;
datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
Lecun = load(fullfile(datax_dir(1).folder,'MSD_lecun.mat'));
loss_all = [];
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
%% plot
figure_width = 16;
total_row = 2;
total_column = 2;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 1.5;
% [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
EMH = 0.15;
EMV = 0.3;
MLR = 0.55;
MBR = 0.8;
[ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.3, MLR, MBR );

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!
%MSL vs distance
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
select_num = 24;
map = jet(select_num);
hold on
for k=1:select_num%length(tau)
    % coarse grain for visualization
    [distance_coarse,MSL_coarse] = coarse_grain(distance{k},MSL_distance{k},50);
    plot(distance_coarse,MSL_coarse,'.-','color',map(k,:))
end
xlim([0.1,10])
xlabel('Distance')
y = ylabel('Mean squared loss');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.1,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[0.1,1,10])
% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e3])
colormap('jet')
c = colorbar('Position', [originalSize(1)+originalSize(3)+0.03, originalSize(2), 0.015, originalSize(4)-0.02],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
% set(T,'Units', 'Normalized', 'Position', [0.5,0.1, 0]);
set(gca, 'Position', originalSize);

%MSL vs contour length
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
select_num = 24;
map = jet(select_num);
hold on
for k=1:select_num%length(tau)
    % coarse grain for visualization
    [distance_coarse,MSL_coarse] = coarse_grain(contour_length{k},MSL_contourlength{k},50);
    plot(distance_coarse,MSL_coarse,'.-','color',map(k,:))
end
xlabel('Contour length')
y = ylabel('Mean squared loss');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.1,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

% % projected landscape fbm test
% load('/import/headnode1/gche4213/Project3/projected_landscape_fbm.mat')
% subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% cla
% plot(C,loss_,'ko')
% hold on
% plot([0.0001,10],[0.001,10].^(1).*exp(1.2),'r','linewidth',linewidth)
% axis([0.01,3,0.1,8])
% xlabel('Distance')
% ylabel('Mean squared loss')
% text(-0.18,1.1,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

%loss spectrum
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
loglog(Lecun.f_loss,Lecun.P1_loss,'K')
xlabel('Frequency (step^{-1})')
ylabel('Loss spectrum')
axis([1e-2,1e3,1e-6,1])
text(-0.18,1.1,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[0.01,1,100])

% 2D landscape hurst exponent
subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
M = csvread('/import/headnode1/gche4213/Project3/box_dimenstion_56.csv');
loglog(exp(M(:,1)),exp(M(:,2)),'ko')
hold on
plot([0.001,10],[0.001,10].^(-1.7).*exp(2.1),'r','linewidth',linewidth)
axis([0.006,1,64,1e5])
xlabel('Size')
ylabel('Number')
text(-0.18,1.1,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

set(gcf, 'PaperPositionMode', 'auto');

% output
print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/hierarchical_landscape.svg','-dsvg','-r300')
function [distance_coarse,MSL_coarse] = coarse_grain(distance,MSL,coarse_num)
% coarse grain for visualization
Dis_coarse = linspace(min(distance),max(distance),coarse_num+1);
for interval = 1:coarse_num
    MSL_coarse(interval) = mean(MSL((distance >= Dis_coarse(interval)) & (distance < Dis_coarse(interval+1))));
end
distance_coarse = movmean(Dis_coarse,2);
distance_coarse = distance_coarse(2:end);
end