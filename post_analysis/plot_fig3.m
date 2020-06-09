clear
close all
%% load data
d = dir('/trained_nets/resnet*');
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
figure_width = 20;
total_row = 1;
total_column = 3;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 0.7;
ylabel_shift = -0.25;
xlabel_shift = -0.2;
TickLength = 0.03;
EMH = 0.1;
EMV = 0.5;
MLR = 1;
MBR = 1;
[figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.4, EMV, 0.3, MLR, MBR );

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!
%MSL vs distance
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
select_num = 24;
map = [0,90,171;71,31,31;255,66,93]./255;
hold on
kkk = 1;
for k=[1,round(select_num/5),select_num]%length(tau)
    % coarse grain for visualization
    [distance_coarse,MSL_coarse] = coarse_grain(distance{k},MSL_distance{k},50);
    plot(distance_coarse,MSL_coarse,'-','color',map(kkk,:),'linewidth',1)
    kkk = kkk + 1;
end
xlim([0.1,10])
x = xlabel('{\Delta}r');
set(x, 'Units', 'Normalized', 'Position', [0.5,xlabel_shift, 0]);
y = ylabel('MSL');
set(y, 'Units', 'Normalized', 'Position', [ylabel_shift, 0.5, 0]);
legend({'t_w=1 step','t_w=5001','t_w=24000'})
legend boxoff
text(-0.2,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[0.1,1,10],'TickLength',[TickLength 0.035])

% leave space for schematic diagram of contour length
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
text(-0.2,1.15,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')

% contour length vs MSD
cc = subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
select_num = 24;
map = [0,90,171;71,31,31;255,66,93]./255;
kkk = 1;
for k = [2,round(select_num/5),select_num]
    loglog(contour_length{k},smoothdata(MSD_forcontour{k},'movmean',[40,400]),'color',map(kkk,:))
kkk = kkk + 1;
end
x = xlabel('Contour length');
set(x, 'Units', 'Normalized', 'Position', [0.5,xlabel_shift, 0]);
y = ylabel('{\Delta}r^2');
set(y, 'Units', 'Normalized', 'Position', [ylabel_shift, 0.5, 0]);
axis([0.08 200 8e-3 100])
legend({'t_w=1 step','t_w=5001','t_w=24000'})
legend boxoff
text(-0.2,1.15,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[0.1,1,10,100],'TickLength',[TickLength 0.035])

% a single contour length vs MSD curve
insect = axes('position',[0.817460317460317 0.558596971434323 0.0806878306878308 0.333155605885265]);
axis(insect)
k = 2;
plot(contour_length{k},MSD_forcontour{k},'color',map(k,:),'linestyle','none','marker','.')
hold on
plot(contour_length{k},smoothdata(MSD_forcontour{k},'movmean',[40,400]),'r:','linewidth',linewidth)
axis([0.08 200 8e-3 100])
legend({'Raw','Smo.'},'location','northwest')
legend boxoff
set(gca,'xscale','log','yscale','log','xtick',[],'ytick',[])

set(gcf, 'PaperPositionMode', 'auto');

% output
print('-painters' ,'fig3.svg','-dsvg','-r300')
function [distance_coarse,MSL_coarse] = coarse_grain(distance,MSL,coarse_num)
% coarse grain for visualization
Dis_coarse = linspace(min(distance),max(distance),coarse_num+1);
for interval = 1:coarse_num
    MSL_coarse(interval) = mean(MSL((distance >= Dis_coarse(interval)) & (distance < Dis_coarse(interval+1))));
end
distance_coarse = movmean(Dis_coarse,2);
distance_coarse = distance_coarse(2:end);
end