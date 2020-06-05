% results are not so consistent
clear
% close all
spatial_epson = 1e-3;
ii = 2;
% fractal
% L = load('/import/headnode1/gche4213/Project3/post_analysis/bigger_fractal_landscape.mat');
% load('/import/headnode1/gche4213/Project3/test_toy/test_toy_fractal.mat')
% x = L.x;
% y = L.y;
% zz = L.zz;
% z = zz{ii};
% landscape_F = griddedInterpolant((repmat(x',1,length(y))-1)*2-1,(repmat(y,length(x),1)-1)*2-1,z,'linear');% may changet to nonlinear interpolation
% look_up_table = -1:spatial_epson:1;
% [xq, yq] = ndgrid(look_up_table);
% landscape = 10-landscape_F(xq, yq);


% convex
load('/import/headnode1/gche4213/Project3/test_toy_gaussian/test_toy_fractal.mat')
look_up_table = -1:spatial_epson:1;
[x,y]=ndgrid(look_up_table);
landscape = 10-10*exp(-(x.^2 + y.^2)./2);


t = (1:length(X{ii}))';
loss = zeros(length(X{ii}),1);
for j=1:length(X{ii})
    [~,x_pos] = min(abs(look_up_table - X{ii}(j)));
    [~,y_pos] = min(abs(look_up_table - Y{ii}(j)));
    loss(j) = landscape(x_pos,y_pos);
end
Trajectory = single([X{ii},Y{ii},t,loss]);
[MSD,Mean_contourlength,tau,MSD_forcontour,MSL_contourlength,contour_length,MSL_distance] = get_contour_lenth_MSD_loss(Trajectory);

figure
linewidth = 1.5;
fontsize = 12;
% avalanche dur distribution
subplot(2,2,1)
dur_all = calculate_avalance_dur(diff(loss));
[a_all, xmin_all, Loglike_all] = plfit(dur_all,'finite');
plplot_selfdefine(dur_all(dur_all>0), xmin_all, a_all);
y = ylabel('$P(D /ge d)$','interpreter','latex');
xlabel('Duration of avalanche');
set(y, 'Units', 'Normalized', 'Position', [-0.24, 0.5, 0]);
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')

% avalanche dur distribution
subplot(2,2,2)
plot(contour_length,smoothdata(MSD_forcontour,'movmean',[40,40]))
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')
xlabel('Contour length')
ylabel('MSD')

subplot(2,2,3)
plot(contour_length,smoothdata(MSL_contourlength,'movmean',[40,40]))
set(gca,'linewidth',linewidth,'fontsize',fontsize,'tickdir','out','xscale','log','yscale','log')
xlabel('Contour length')
ylabel('MSL')

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
        fprintf('(PLPLOT) Error: x must contain only reals or only integers./n');
        h = [];
        return;
end
end