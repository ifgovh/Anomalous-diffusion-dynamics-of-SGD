% Simple demo on the use of MF_BS_tool.
% Does wavelet leader based multifractal analysis and illustrates basic
% features.
%
% Roberto Leonarduzzi
% 2016-07-04

% Add root folder of toolbox to path (subfolders are added automatically)
addpath E:\mf_bs_tool


%% Select example data

MFprocess = 3;


        data = h5read('model_500.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5','/train_loss');
        data = reshape(data(randperm(numel(data))),51,51);
        N = length (data);
        figure (11111); clf
        surf (data)
        TSTR = 'A realization of 2d compound poisson cascade with log-normal multipliers (c_1=0.0125, c_2=-0.025, N=1024x1024)';
        %data = data (1 : 21, 1 : 21);

title(TSTR);
grid on;



%% Setup analysis

% Create an object of class MF_BS_tool_inter to perform analysis
mf_obj = MF_BS_tool_inter;

% All analysis parameters are properties of this class that can be accessed
% with operator '.'

% Select order of Daubechies wavelet:
mf_obj.nwt = 3;

% Select which multiresolution quantities are used:
mf_obj.method_mrq = [1 2];    % 1: Wavelet coefficients, 2: wavelet leaders

% Set up scaling range:
mf_obj.j1      = 1;
mf_obj.j2      = 3;

% Use a convenient set of orders q
mf_obj.q       = build_q_log (0.01, 5, 10);

% Compute the first three log-cumulants
mf_obj.cum     = 3;

% Indicate pseudo-fractional-integration order
mf_obj.gamint = 0;

% Indicate figure number to be used by the tool
mf_obj.fig_num = 100;

% Set up versbosity level: show figures:
mf_obj.verbosity = 1;   % Number xy, with x,y either 0 or 1
                        %    x = 1: display figures
                        %    y = 1: display tables with estimates

%% Perform analysis

% The input can be a 1d or 2d array.
mf_obj.analyze (data);

%% Access results

% Get estimates for wavelet coefficients and leaders:
t_dwt = mf_obj.est.DWT.t;  % (size nest x 1)
t_lwt = mf_obj.est.LWT.t;  % (size nest x 1)

% Get logscale diagrams for wavelet coefficients and leaders:
lsd_dwt = mf_obj.logstat.DWT.est;  % (size nest x nj)
lsd_lwt = mf_obj.logstat.LWT.est;  % (size nest x nj)

% The nest estimates are ordered as [zeta(q), D(q), h(q), c_m]
% Each one can be accessed individually as:

zid = mf_obj.get_zid ();  % Indices of zeta(q)
hid = mf_obj.get_hid ();  % Indices of h(q)
Did = mf_obj.get_Did ();  % Indices of D(q)
cid = mf_obj.get_cid ();  % Indices of c_p

zq = mf_obj.est.LWT.t(zid);  % Estimates of zeta(q)
hq = mf_obj.est.LWT.t(hid);  % Estimates of h(q)
Dq = mf_obj.est.LWT.t(Did);  % Estimates of D(q)
cp = mf_obj.est.LWT.t(cid);  % Estimates of c_p

% Structure functions:
sf_zq  = mf_obj.logstat.LWT.est(zid, :);  % Structure functions log_2 S(q,j)
sf_hq  = mf_obj.logstat.LWT.est(hid, :);  % "Structure functions" log_2 U(q,j)
sf_Dq  = mf_obj.logstat.LWT.est(Did, :);  % "Structure functions" log_2 V(q,j)
sf_cum = mf_obj.logstat.LWT.est(cid, :);  % Cumulants C_m(j)

%% Redo analysis with different scaling range

% The analysis can be efficiently redone with a different scaling range, without recomputing
% the multiresolution quantities and structure functions, by reusing the
% stored logstat property.

% Change scaling range, report analysis in different figures:
if MFprocess == 3
    mf_obj.j1 = 1;
    mf_obj.j2 = -6;  % Negative numbers are counted from the largest scale.

else
    mf_obj.j1 = 5;
    mf_obj.j2 = 10;
end
mf_obj.fig_num = 200;

% Redo analysis using the stored logstat structure:
mf_obj.analyze ()

% Redo analysis providing a logstat structure explicitly:
% mf_obj.analyze (mf_obj.logstat)
