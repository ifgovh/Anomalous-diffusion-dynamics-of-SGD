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

% eta/batch_size ratio plot
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%select_num = length(MSD);
map = [0,90,171;71,31,31;255,66,93]./255;

xlabel('\eta/B')
y = ylabel('\alpha in the first regime');
set(y, 'Units', 'Normalized', 'Position', [ylable_shift, 0.5, 0]);
text(-0.18,1.15,'(a)','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','xtick',[1,10,100,1000],'TickLength',[TickLength 0.035])

%plot(lb_ratio,alpha_ls,'linewidth',linewidth,'o');
plot(alpha_lb_combined.alpha_ls1,alpha_lb_combined.lb1,'-o')
hold on;
plot(alpha_lb_combined.alpha_ls2,alpha_lb_combined.lb2,'-o')
hold on;
plot(alpha_lb_combined.alpha_ls3,alpha_lb_combined.lb3,'-o')
hold on;
plot(alpha_lb_combined.alpha_ls4,alpha_lb_combined.lb4,'-o')
hold on;
plot(alpha_lb_combined.alpha_ls5,alpha_lb_combined.lb5,'o-')
legend({'SC 14','SC 14 G','SC 14 NB','NS 14 NB','SC MN'},'Location','best')
legend boxoff

set(gcf, 'PaperPositionMode', 'auto');

% output
%saveas(gcf,sprintf('test_noavearge_MSD_gradient_%d.fig',ii))
saveas(gcf,sprintf('lr_batch_ratio_combined.fig'))