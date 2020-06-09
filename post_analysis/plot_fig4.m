clear
close all
figure_width = 20;
total_row = 2;
total_column = 3;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 0.7; 
TickLength = 0.03;
% [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
EMH = 0.1;
EMV = 0.3;
MLR = 0.65;
MBR = 1.1;
[ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.5, EMV, 0.3, MLR, MBR );

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!    
% toy model trajectory and landscape
cc = subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
load('/simplified_model/test_toy_fractal.mat','X','Y','land_ind')
spatial_epson = 1e-3;
L = load('/post_analysis/bigger_fractal_landscape.mat');
x = L.x;
y = L.y;
zz = L.zz;
rd = 2;%83;%277;
z = zz{land_ind(rd)};
bb=find(Y{rd}>-0.3385,1);
bb = bb + 200;
landscape_F = griddedInterpolant((repmat(x',1,length(y))-1)*2-1,(repmat(y,length(x),1)-1)*2-1,z,'linear');% may changet to nonlinear interpolation
look_up_table = -1:spatial_epson:1;
[xq, yq] = ndgrid(look_up_table);
landscape = 10-landscape_F(xq, yq);

hold on
contour(look_up_table,look_up_table,landscape)
Traj = plot(X{rd}(bb:end),Y{rd}(bb:end),'k-','linewidth',linewidth*0.7);
start = plot(X{rd}(bb),Y{rd}(bb),'rx');
end_point = plot(X{rd}(end),Y{rd}(end),'ro');
legend([Traj,start,end_point],{'Traj.','Start','End'})
% xlabel('X')
% ylabel('Y')
% colorbar
originalSize = get(gca, 'Position');
% caxis([])
colormap(cc,'default')
c = colorbar('Position', [originalSize(1)+originalSize(3)+0.01  originalSize(2)  0.007 originalSize(4)],'location','east');
T = title(c,'Altitude','fontsize',fontsize);
set(gca, 'Position', originalSize);
text(-0.18,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

% % MSD of toy model calculating from the whole path
% subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
%     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% % subaxis(total_row,total_column,2,3,'SpacingHoriz',SH,...
% %     'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
% % h = axes('Position',[0.611164462809917 0.231930526462786 0.366032210834553 0.0793650793650791],...
% %     'Tag','subaxis');
% % set(h,'box','on');
% %h=axes('position',[x1 1-y2 x2-x1 y2-y1]);
% % set(h,'units',get(gcf,'defaultaxesunits'));
t = (1:1000+1)';
% [MSD_total,tau_total] = get_MSD([X{rd}(bb:end),Y{rd}(bb:end),t(bb:end)]);
% loglog(tau_total,MSD_total,'k','linewidth',linewidth)
% ylabel('{\Delta}r^2(\tau)')
% xlabel('\tau (step)')
% text(-0.18,1.15,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
% set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')%,'ytick',[1e-2,1])

% MSD in the initial phase
subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
% h=axes('position',[0.611164462809917 0.154139784946237 0.366032210834553 0.0793650793650792]);
% set(h,'box','on');
%h=axes('position',[x1 1-y2 x2-x1 y2-y1]);
% set(h,'units',get(gcf,'defaultaxesunits'));
% set(h,'tag','subaxis');
% aa=find(X{rd}<-0.3284,1);
aa=find(X{rd}<-0.5,1);
[MSD_initial,tau_initial] = get_MSD([X{rd}(bb:aa),Y{rd}(bb:aa),t(bb:aa)]);
loglog(tau_initial,MSD_initial,'k-','linewidth',linewidth)
F1 = plot(tau_initial(round(numel(tau_initial)/5)),MSD_initial(round(numel(tau_initial)/5)),'ks');

[MSD_end,tau_end] = get_MSD([X{rd}(aa:end),Y{rd}(aa:end),t(aa:end)]);
loglog(tau_end,MSD_end,'k-','linewidth',linewidth)
F2 = plot(tau_end(round(numel(tau_initial)/5)),MSD_end(round(numel(tau_initial)/5)),'ko');

legend([F1,F2],{'R1 fractal','R2 fractal'})
legend boxoff
xlabel('\tau (step)')
ylabel('{\Delta}r^2(\tau)')
ylim([0.00025,2.5])
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','ytick',[1e-3,1e-2,1e-1,1],'TickLength',[TickLength 0.035])
text(-0.18,1.15,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')

% gradient distribution
subaxis(total_row,total_column,3,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);

% fractal landscape
load('/simplified_model/test_toy_fractal.mat')
for ii = 1:length(gradient_x)
    G_f(:,ii) = [gradient_x{ii},gradient_y{ii}];
end
% Convex landscape
load('/simplified_model/test_toy_gaussian.mat')
for ii = 1:length(d)
    G_c(:,ii) = [gradient_x{ii},gradient_y{ii}];
end
% randomly shuffled landscape
load('/simplified_model/test_toy_shuffle_fractal.mat')
for ii = 1:length(d)
    G_r(:,ii) = [gradient_x{ii},gradient_y{ii}];
end

hold on
h1 = histogram(G_f,linspace(-5,5,51),'normalization','pdf','edgealpha',1,'facealpha',1,'facecolor',[151,151,151]./255);
h2 = histogram(G_c,linspace(-5,5,51),'normalization','pdf','edgealpha',0.7,'facealpha',0.7,'facecolor',[36,169,225]./255);
h3 = histogram(G_r,linspace(-5,5,51),'normalization','pdf','edgealpha',0.5,'facealpha',0.5,'facecolor',[178,18,88]./255);
legend([h1,h2,h3],{'Fractal','Convex','Random'})
legend boxoff
xlim([-5,5])
xlabel('Gradient')
ylabel('PDF')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')%,'ytick',[1e-2,1e-1])
text(-0.18,1.15,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')

% % log-log
% insect = axes('position',[0.900263852242744 0.817734053450724 0.0880735994192455 0.149985821741288]);
% axis(insect);box on;
% F_f = fitdist(G_f(35,:)','stable');
% plot(linspace(0,10,1000),pdf(F_f,linspace(0,10,1000)),'r-')
% % axis([1,1e3,1e-2,1e2])
% set(gca,'xscale','log','yscale','log','tickdir','out') 

% color-encoded step size
load('/simplified_model/test_toy_fractal.mat','X','Y','land_ind')
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on
% one should specify the zoom-in area based on one's own results
contour(look_up_table(1151:1302),look_up_table(852:1002),landscape(852:1002,1151:1302),30)
cc = find(X{rd}<0.4,1);
dd = find(X{rd}<0.14,1);
step_size = sqrt(diff(X{rd}(cc:dd)).^2+diff(Y{rd}(cc:dd)).^2);
[C,~,ic] = unique(step_size);
map = copper(length(C));
hold on
for k = cc:dd-1
plot([X{rd}(k),X{rd}(k+1)],[Y{rd}(k),Y{rd}(k+1)],'-','linewidth',linewidth*4,'color',map(ic(k-cc+1),:));
end
axis([0.15,0.3,-0.15,0])
text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')

% random landscape
subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
load('/simplified_model/test_toy_shuffle_fractal.mat','X','Y','landscape_pool')
rd = 1;
hold on
contour(look_up_table,look_up_table,landscape_pool{rd})
plot(X{rd}(1:200),Y{rd}(1:200),'k-','linewidth',linewidth*0.7);
plot(X{rd}(1),Y{rd}(1),'rx');
plot(X{rd}(200),Y{rd}(200),'ro');
text(-0.18,1.15,'e','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')


% MSD in surrogate cases
subaxis(total_row,total_column,3,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
hold on

% convex
convex_data = load('/simplified_model/test_toy_gaussian.mat');
[MSD_convex,tau_convex] = get_MSD([convex_data.X{1}(1:144),convex_data.Y{1}(1:144),(1:144)']);
loglog(tau_convex,MSD_convex,'k-','linewidth',linewidth)
C = plot(tau_convex(round(numel(tau_initial)/5)),MSD_convex(round(numel(tau_initial)/5)),'k^');

% random
rand_data = load('/simplified_model/test_toy_shuffle_fractal.mat');
[MSD_rand,tau_rand] = get_MSD([rand_data.X{1},rand_data.Y{1},t]);
loglog(tau_rand,MSD_rand,'k-','linewidth',linewidth)
R = plot(tau_rand(round(numel(tau_initial)/5)),MSD_rand(round(numel(tau_initial)/5)),'kx');

legend([C,R],{'Convex','Random'})
legend boxoff
xlabel('\tau (step)')
ylabel('{\Delta}r^2(\tau)')
ylim([0.00025,2.5])
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log','ytick',[1e-3,1e-2,1e-1,1],'TickLength',[TickLength 0.035])
text(-0.18,1.15,'f','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')



set(gcf, 'PaperPositionMode', 'auto');

% output
print('-painters' ,'fig4.svg','-dsvg','-r300')
