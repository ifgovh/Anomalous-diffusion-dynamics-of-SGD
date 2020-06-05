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
        Contour_length_all{part} = L.Contour_length_all;
        loss_all = [loss_all;L.delta_train_loss];
    catch
        loss{part} = nan;
        test_loss{part} = nan;
        delta_train_loss{part} = nan;
        delta_test_loss{part} = nan;        
        Displacement_all{part} = nan;
        Contour_length_all{part} = nan;
    end
end

load('/import/headnode1/gche4213/Project3/trans_ex_avalan.mat')
ii = 2;
%%
figure_width = 16;
total_row = 3;
total_column = 2;
% uniform FontSize and linewidth
fontsize = 10;
linewidth = 1.5;
% [ verti_length, verti_dis, hori_dis ] = get_details_for_subaxis( total_row, total_column, hori_length, edge_multiplyer_h, inter_multiplyer_h, edge_multiplyer_v, inter_multiplyer_v )
EMH = 0.15;
EMV = 0.45;
MLR = 0.8;
MBR = 0.8;
[ figure_hight, SV, SH, MT, MB, ML, MR ] = get_details_for_subaxis(total_row, total_column, figure_width, EMH, 0.4, EMV, 0.3, MLR, MBR );

figure('NumberTitle','off','name', 'trapping', 'units', 'centimeters', ...
    'color','w', 'position', [0, 0, figure_width, figure_hight], ...
    'PaperSize', [figure_width, figure_hight]); % this is the trick!

% plot avalanche enviroment
load('/import/headnode1/gche4213/Project3/trans_ex_avalan.mat')
% L vs contour length
subaxis(total_row,total_column,1,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
part = 1;
plot(contour_len{2}{part},seg_loss{2}{part},'color','k')
hold on
plot(contour_len{2}{end},seg_loss{2}{end},'color','r')
y = ylabel('Loss');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
% xlabel('Contour length');
text(-0.18,1.15,'a','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')
%\delta L vs contour length
subaxis(total_row,total_column,1,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
plot(contour_len{2}{part},seg_delta_loss{2}{part},'color','k')
hold on
plot(contour_len{2}{end},seg_delta_loss{2}{end},'color','r')
y = ylabel('|{\Delta}Loss|');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
x = xlabel('Contour length');
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'b','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

% \delta Loss vs t
subaxis(total_row,total_column,1,3,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
delta_loss_temp = diff(loss_all{ii});
hold on
for k=1%:length(tau)
    %plot(1:length(delta_train_loss{k}),abs(delta_train_loss{k}),'color',map(k,:))
    plot(abs(delta_loss_temp(1:80)),'k.-')
end

x = xlabel('Time (step)');
y = ylabel('|{\Delta}Loss|');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'c','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')

%\delta L distribution
cc = subaxis(total_row,total_column,2,1,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
select_num = 24;
map = jet(select_num);
hold on
for k=1:select_num%length(tau)
    [N,edges] = histcounts(delta_train_loss{k},'normalization','pdf');
    plot(edges(1:end-1),N,'o-','color',map(k,:))    
end
ytickangle(45)
x = xlabel('{\Delta}Loss');
y = ylabel('PDF');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
set(x, 'Units', 'Normalized', 'Position', [0.5, -0.15, 0]);
text(-0.18,1.15,'d','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out')


% avalanche dur distribution
subaxis(total_row,total_column,2,2,'SpacingHoriz',SH,...
    'SpacingVert',SV,'MR',MR,'ML',ML,'MT',MT,'MB',MB);
dur_all = calculate_avalance_dur(diff(loss_all{ii}));
[a_all, xmin_all, Loglike_all] = plfit(dur_all,'finite');    
plplot_selfdefine(dur_all(dur_all>0), xmin_all, a_all);
y = ylabel('$P(D \ge d)$','interpreter','latex');
xlabel('Duration of avalanche');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
text(-0.18,1.15,'e','fontsize',fontsize,'Units', 'Normalized', 'FontWeight','bold','VerticalAlignment', 'Top')
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')


set(gcf, 'PaperPositionMode', 'auto');

% output
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/avalanche_environment2.svg','-dsvg','-r300')
% print('-painters' ,'/import/headnode1/gche4213/Project3/outputfigures/avalanche_environment.pdf','-dpdf','-r300')

function dur = calculate_avalance_dur(L)
L = abs(L);
SD = std(L);
bool_vec = L > SD;
CC = bwconncomp(bool_vec);
dur = cellfun(@length,CC.PixelIdxList);
end


function h=plplot_selfdefine(x, xmin, alpha)
% PLPLOT visualizes a power-law distributional model with empirical data.
%    Source: http://www.santafe.edu/~aaronc/powerlaws/
% 
%    PLPLOT(x, xmin, alpha) plots (on log axes) the data contained in x 
%    and a power-law distribution of the form p(x) ~ x^-alpha for 
%    x >= xmin. For additional customization, PLPLOT returns a pair of 
%    handles, one to the empirical and one to the fitted data series. By 
%    default, the empirical data is plotted as 'bo' and the fitted form is
%    plotted as 'k--'. PLPLOT automatically detects whether x is composed 
%    of real or integer values, and applies the appropriate plotting 
%    method. For discrete data, if min(x) > 50, PLFIT uses the continuous 
%    approximation, which is a reliable in this regime.
%
%    Example:
%       xmin  = 5;
%       alpha = 2.5;
%       x = xmin.*(1-rand(10000,1)).^(-1/(alpha-1));
%       h = plplot(x,xmin,alpha);
%
%    For more information, try 'type plplot'
%
%    See also PLFIT, PLVAR, PLPVA

% Version 1.0   (2008 February)
% Copyright (C) 2008-2011 Aaron Clauset (Santa Fe Institute)
% Distributed under GPL 2.0
% http://www.gnu.org/copyleft/gpl.html
% PLFIT comes with ABSOLUTELY NO WARRANTY
% 
% No notes
% 

% reshape input vector
x = reshape(x,numel(x),1);
% initialize storage for output handles
h = zeros(2,1);

% select method (discrete or continuous) for plotting
if     isempty(setdiff(x,floor(x))), f_dattype = 'INTS';
elseif isreal(x),    f_dattype = 'REAL';
else                 f_dattype = 'UNKN';
end;
if strcmp(f_dattype,'INTS') && min(x) > 50,
    f_dattype = 'REAL';
end;

% estimate xmin and alpha, accordingly
switch f_dattype,
    
    case 'REAL',
        n = length(x);
        c = [sort(x) (n:-1:1)'./n];
        q = sort(x(x>=xmin));
        cf = [q (q./xmin).^(1-alpha)];
        cf(:,2) = cf(:,2) .* c(find(c(:,1)>=xmin,1,'first'),2);

%         figure;
        h(1) = loglog(c(:,1),c(:,2),'bo','MarkerSize',2,'MarkerFaceColor',[1 1 1]); hold on;
        h(2) = loglog(cf(:,1),cf(:,2),'k-'); hold off;
        xr  = [10.^floor(log10(min(x))) 10.^ceil(log10(max(x)))];
        xrt = (round(log10(xr(1))):2:round(log10(xr(2))));
        if length(xrt)<4, xrt = (round(log10(xr(1))):1:round(log10(xr(2)))); end;
        yr  = [10.^floor(log10(1/n)) 1];
        yrt = (round(log10(yr(1))):2:round(log10(yr(2))));
        if length(yrt)<4, yrt = (round(log10(yr(1))):1:round(log10(yr(2)))); end;
        set(gca,'XLim',xr);
        set(gca,'YLim',yr);
        

    case 'INTS',
        n = length(x);        
        q = unique(x);
        c = hist(x,q)'./n;
        c = [[q; q(end)+1] 1-[0; cumsum(c)]]; c(c(:,2)<10^-10,:) = [];
        cf = ((xmin:q(end))'.^-alpha)./(zeta(alpha) - sum((1:xmin-1).^-alpha));
        cf = [(xmin:q(end)+1)' 1-[0; cumsum(cf)]];
        cf(:,2) = cf(:,2) .* c(c(:,1)==xmin,2);

%         figure;
        h(1) = loglog(c(:,1),c(:,2),'ko','MarkerSize',3,'MarkerFaceColor',[1 1 1]); hold on;
        %h(2) = loglog(cf(:,1),cf(:,2),'r--','MarkerSize',1); hold off;
        xr  = [10.^floor(log10(min(x))) 10.^ceil(log10(max(x)))];
        xrt = (round(log10(xr(1))):2:round(log10(xr(2))));
        if length(xrt)<4, xrt = (round(log10(xr(1))):1:round(log10(xr(2)))); end;
        yr  = [10.^floor(log10(1/n)) 1];
        yrt = (round(log10(yr(1))):2:round(log10(yr(2))));
        if length(yrt)<4, yrt = (round(log10(yr(1))):1:round(log10(yr(2)))); end;
        set(gca,'XLim',xr);
        set(gca,'YLim',yr);        

    otherwise,
        fprintf('(PLPLOT) Error: x must contain only reals or only integers.\n');
        h = [];
        return;
end
end