% plot figures of Proj3
clear
close all
%% load data
d = dir('/import/headnode1/gche4213/Project3/*net*');
ii = 2;

datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
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
        loss_all = [loss_all;L.delta_train_loss];
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
EMV = 0.4;
MLR = 0.55;
MBR = 0.9;
[ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.3, MLR, MBR );

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!


% leave space for schematic diagram of contour length
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
text(-0.16,1.12,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')

% contour length vs MSD
cc = subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = jet(select_num);
for k=2:select_num%length(tau)
    loglog(contour_length{k},smoothdata(MSD_forcontour{k},'movmean',[40,400]),'color',map(k,:),'linewidth',linewidth)
end
x = xlabel('{\Delta}s');
y = ylabel('{\Delta}r^2');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
axis([0.08 200 8e-3 100])
text(-0.16,1.12,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[0.1,1,10,100])

% colorbar
originalSize = get(gca, 'Position');
caxis([1,1 + (select_num-1)*1e3])
colormap(cc,'jet')
c = colorbar('Position', [originalSize(1) + originalSize(3) + 0.02  originalSize(2)  0.015 originalSize(4)],'location','east');
T = title(c,'t_w (step)','fontsize',fontsize);
% set(T,'Units', 'Normalized', 'Position', [0,0.5, 5]);
set(gca, 'Position', originalSize);

% a single contour length vs MSD curve
insect = axes('position',[0.0965203491839572 0.302335839032963 0.17 0.17]);
axis(insect)
k = 2;
plot(contour_length{k},MSD_forcontour{k},'color',map(k,:),'linestyle','none','marker','.')
hold on
plot(contour_length{k},smoothdata(MSD_forcontour{k},'movmean',[40,400]),'r:','linewidth',linewidth)
axis([0.08 200 8e-3 100])
legend({'Raw','Smo.'},'location','northwest')
legend boxoff
% x = xlabel('Contour length');
% set(x, 'Units', 'Normalized', 'Position', [0.5, -0.1, 0]);
% y = ylabel('{\Delta}r^2');
% set(y, 'Units', 'Normalized', 'Position', [-0.22, 0.5, 0]);
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

% transverse excursion
ii = 10;
load(['/import/headnode1/gche4213/Project3/trans_ex_avalan_',num2str(ii),'.mat'])
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
for part = 1:select_num%length(end_end_dis2{ii})
    plot(end_end_dis2{ii}{part},transversed_excursion{ii}{part},'color',map(part,:),'linestyle','none','marker','.')
end
x = xlabel('{\Delta}r^2');
y = ylabel('Transverse excursion^2');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
% axis([])
text(-0.16,1.12,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

% contour length
cc = subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
for k=1:length(tau)
    loglog(tau{k},Mean_contourlength{k},'color',map(k,:),'linewidth',linewidth)
end
axis([1,1e3,0.2, 200])
x = xlabel('\tau (step)');
y = ylabel('{\Delta}s');
set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.16,1.12,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

% a single contour length curve
insect = axes('position',[0.622640352043925 0.302074111294113 0.17 0.17]);
axis(insect)
k = 1;
loglog(tau{k},Mean_contourlength{k},'color',map(k,:))
axis([1,1e3,0.2, 200])
% x = xlabel('\tau');
% set(x, 'Units', 'Normalized', 'Position', [0.5, -0.1, 0]);
% y = ylabel('{\Delta}r^2(\tau)');
% set(y, 'Units', 'Normalized', 'Position', [-0.22, 0.5, 0]);
set(gca,'tickdir','out','xscale','log','yscale','log','xtick',[],'ytick',[])

set(gcf, 'PaperPositionMode', 'auto');

% output
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/avalanche_contour4.svg','-dsvg','-r300')


