% plot figures of Proj3
clear
close all
%% load data
% all data are from /Project3/MSD_exponent_DNN.xlsx
keySet = {'14_128_0','14_512_0','14_1024_0','14_128_1','14_512_1','14_1024_1',...
    '20_128_0','20_512_0','20_1024_0','20_128_1','20_512_1','20_1024_1',...
    '56_128_0','56_512_0','56_1024_0','56_128_1','56_512_1','56_1024_1',...
    '110_128_0','110_256_0','110_512_0','110_128_1','110_256_1','110_512_1'};    
valueSet_exp1 = [1.358974359,1.221590909,1.387755102,1.315384615,1.404958678,1.214689266,...
1.230769231,1.297709924,1.31372549,1.227436823,1.36,1.36,...
1.111111111,1.1,1.033333333,1.157894737,1.172413793,1.259259259,...
1.012048193,1.172413793,1.35,1.215753425,1.181818182,1.416666667
];
valueSet_exp2 = [0.7,0.69,0.69,0.726666667,0.7,0.782608696,0.649717514,0.683333333,0.693333333,0.733333333,...
    0.703333333,0.6,0.766666667,0.716666667,0.7,0.684210526,0.75,0.733333333,0.7,0.745762712,0.733333333,0.8,...
    0.88372093,0.67357513];
valueSet_tau = [80,50,65,180,100,70,140,60,100,150,100,45,200,125,133,70,70,70,300,400,300,220,110,100];
M_exp1 = containers.Map(keySet,valueSet_exp1);
M_exp2 = containers.Map(keySet,valueSet_exp2);
M_tau = containers.Map(keySet,valueSet_tau);
%% plot 
figure_width = 16;
total_row = 3;
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

%% depth vs exponent
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% colormap
brewermap('Oranges');
map = brewermap(5);
brewermap('Blues');
map2 = brewermap(5);
hold on
%% batch size 128
% with short cut
plot([14,20,56,110],[M_exp1('14_128_1'),M_exp1('20_128_1'),M_exp1('56_128_1'),M_exp1('110_128_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth)
% without short cut 
a = plot([14,20,56,110],[M_exp1('14_128_0'),M_exp1('20_128_0'),M_exp1('56_128_0'),M_exp1('110_128_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth);
%% batch size 512
% with short cut
plot([14,20,56,110],[M_exp1('14_512_1'),M_exp1('20_512_1'),M_exp1('56_512_1'),M_exp1('110_512_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth)
% without short cut 
b = plot([14,20,56,110],[M_exp1('14_512_0'),M_exp1('20_512_0'),M_exp1('56_512_0'),M_exp1('110_512_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth);
%% batch size 1024
% with short cut
plot([14,20,56],[M_exp1('14_1024_1'),M_exp1('20_1024_1'),M_exp1('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth)
% without short cut 
c = plot([14,20,56],[M_exp1('14_1024_0'),M_exp1('20_1024_0'),M_exp1('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth);
legend([a,b,c],{'no SC 128','no SC 512','no SC 1024'})
legend boxoff

y = ylabel('MSD exponent 1');
% set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.1,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

%% depth vs exponent 2
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%% batch size 128
% with short cut
a = plot([14,20,56,110],[M_exp2('14_128_1'),M_exp2('20_128_1'),M_exp2('56_128_1'),M_exp2('110_128_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth);
% without short cut 
plot([14,20,56,110],[M_exp2('14_128_0'),M_exp2('20_128_0'),M_exp2('56_128_0'),M_exp2('110_128_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth)
%% batch size 512
% with short cut
b = plot([14,20,56,110],[M_exp2('14_512_1'),M_exp2('20_512_1'),M_exp2('56_512_1'),M_exp2('110_512_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth);
% without short cut 
plot([14,20,56,110],[M_exp2('14_512_0'),M_exp2('20_512_0'),M_exp2('56_512_0'),M_exp2('110_512_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth)
%% batch size 1024
% with short cut
c = plot([14,20,56],[M_exp2('14_1024_1'),M_exp2('20_1024_1'),M_exp2('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth);
% without short cut 
plot([14,20,56],[M_exp2('14_1024_0'),M_exp2('20_1024_0'),M_exp2('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth)
legend([a,b,c],{'SC 128','SC 512','SC 1024'})
legend boxoff
y = ylabel('MSD exponent 2');
% set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.1,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

%% depth vs \tau 0 
subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%% batch size 128
% with short cut
plot([14,20,56,110],[M_tau('14_128_1'),M_tau('20_128_1'),M_tau('56_128_1'),M_tau('110_128_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth)
% without short cut 
plot([14,20,56,110],[M_tau('14_128_0'),M_tau('20_128_0'),M_tau('56_128_0'),M_tau('110_128_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth);
%% batch size 512
% with short cut
plot([14,20,56,110],[M_tau('14_512_1'),M_tau('20_512_1'),M_tau('56_512_1'),M_tau('110_512_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth)
% without short cut 
plot([14,20,56,110],[M_tau('14_512_0'),M_tau('20_512_0'),M_tau('56_512_0'),M_tau('110_512_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth);
%% batch size 1024
% with short cut
plot([14,20,56],[M_tau('14_1024_1'),M_tau('20_1024_1'),M_tau('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth)
% without short cut 
plot([14,20,56],[M_tau('14_1024_0'),M_tau('20_1024_0'),M_tau('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth);

y = ylabel('\tau_0');
xlabel('Depth')
% set(y, 'Units', 'Normalized', 'Position', [-0.18, 0.5, 0]);
text(-0.18,1.1,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')


%% batch vs exponent1
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%% depth 14
% with short cut
a = plot([128,512,1024],[M_exp1('14_128_1'),M_exp1('14_512_1'),M_exp1('14_1024_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth);
% without short cut 
plot([128,512,1024],[M_exp1('14_128_0'),M_exp1('14_512_0'),M_exp1('14_1024_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth)
%% depth 20
% with short cut
b = plot([128,512,1024],[M_exp1('20_128_1'),M_exp1('20_512_1'),M_exp1('20_1024_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth);
% without short cut 
plot([128,512,1024],[M_exp1('20_128_0'),M_exp1('20_512_0'),M_exp1('20_1024_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth)
%% depth 56
% with short cut
c = plot([128,512,1024],[M_exp1('56_128_1'),M_exp1('56_512_1'),M_exp1('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth);
% without short cut 
plot([128,512,1024],[M_exp1('56_128_0'),M_exp1('56_512_0'),M_exp1('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth)
%% depth 110
% with short cut
d = plot([128,256,512],[M_exp1('110_128_1'),M_exp1('110_256_1'),M_exp1('110_512_1')],'marker','o','linestyle','-','color',map(5,:),'linewidth',linewidth);
% without short cut 
plot([128,256,512],[M_exp1('110_128_0'),M_exp1('110_256_0'),M_exp1('110_512_0')],'marker','o','linestyle','-','color',map2(5,:),'linewidth',linewidth)
ylim([1,1.6])
legend([a,b,c,d],{'SC 14','SC 20','SC 56','SC 110'})
legend boxoff
text(-0.18,1.1,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

%% batch vs exponent2
subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%% depth 14
% with short cut
plot([128,512,1024],[M_exp2('14_128_1'),M_exp2('14_512_1'),M_exp2('14_1024_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth)
% without short cut 
a = plot([128,512,1024],[M_exp2('14_128_0'),M_exp2('14_512_0'),M_exp2('14_1024_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth);
%% depth 20
% with short cut
plot([128,512,1024],[M_exp2('20_128_1'),M_exp2('20_512_1'),M_exp2('20_1024_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth)
% without short cut 
b = plot([128,512,1024],[M_exp2('20_128_0'),M_exp2('20_512_0'),M_exp2('20_1024_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth);
%% depth 56
% with short cut
plot([128,512,1024],[M_exp2('56_128_1'),M_exp2('56_512_1'),M_exp2('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth)
% without short cut 
c = plot([128,512,1024],[M_exp2('56_128_0'),M_exp2('56_512_0'),M_exp2('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth);
%% depth 110
% with short cut
plot([128,256,512],[M_exp2('110_128_1'),M_exp2('110_256_1'),M_exp2('110_512_1')],'marker','o','linestyle','-','color',map(5,:),'linewidth',linewidth)
% without short cut 
d = plot([128,256,512],[M_exp2('110_128_0'),M_exp2('110_256_0'),M_exp2('110_512_0')],'marker','o','linestyle','-','color',map2(5,:),'linewidth',linewidth);

legend([a,b,c,d],{'no SC 14','no SC 20','no SC 56','no SC 110'})
legend boxoff

text(-0.18,1.1,'e','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

%% batch vs \tau 0
subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
%% depth 14
% with short cut
plot([128,512,1024],[M_tau('14_128_1'),M_tau('14_512_1'),M_tau('14_1024_1')],'marker','o','linestyle','-','color',map(2,:),'linewidth',linewidth)
% without short cut 
plot([128,512,1024],[M_tau('14_128_0'),M_tau('14_512_0'),M_tau('14_1024_0')],'marker','o','linestyle','-','color',map2(2,:),'linewidth',linewidth)
%% depth 20
% with short cut
plot([128,512,1024],[M_tau('20_128_1'),M_tau('20_512_1'),M_tau('20_1024_1')],'marker','o','linestyle','-','color',map(3,:),'linewidth',linewidth)
% without short cut 
plot([128,512,1024],[M_tau('20_128_0'),M_tau('20_512_0'),M_tau('20_1024_0')],'marker','o','linestyle','-','color',map2(3,:),'linewidth',linewidth)
%% depth 56
% with short cut
plot([128,512,1024],[M_tau('56_128_1'),M_tau('56_512_1'),M_tau('56_1024_1')],'marker','o','linestyle','-','color',map(4,:),'linewidth',linewidth)
% without short cut 
plot([128,512,1024],[M_tau('56_128_0'),M_tau('56_512_0'),M_tau('56_1024_0')],'marker','o','linestyle','-','color',map2(4,:),'linewidth',linewidth)
%% depth 110
% with short cut
plot([128,256,512],[M_tau('110_128_1'),M_tau('110_256_1'),M_tau('110_512_1')],'marker','o','linestyle','-','color',map(5,:),'linewidth',linewidth)
% without short cut 
plot([128,256,512],[M_tau('110_128_0'),M_tau('110_256_0'),M_tau('110_512_0')],'marker','o','linestyle','-','color',map2(5,:),'linewidth',linewidth)

xlabel('Batch size')
text(-0.18,1.1,'f','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

set(gcf, 'PaperPositionMode', 'auto');

% output
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/depth_batch_size.svg','-dsvg','-r300')
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/depth_batch_size.pdf','-dpdf','-r300')
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/depth_batch_size.eps','-depsc','-r300')
