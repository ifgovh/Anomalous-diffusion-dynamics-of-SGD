clear
% % fractal landscape
% load('X:\Project3\test_toy\test_toy_fractal.mat')
% d = dir('X:\Project3\test_toy\good_figures\*jpg');
% Convex landscape
load('X:\Project3\test_toy_gaussian\test_toy_fractal.mat')
d = dir('X:\Project3\test_toy_gaussian\*jpg');
% % randomly shuffled landscape
% load('X:\Project3\test_toy_shuffle\test_toy_shuffle_fractal.mat')
% d = dir('X:\Project3\test_toy_shuffle\*jpg');
for ii = 1:length(d)
    G(:,ii) = [gradient_x{str2num(d(ii).name(6:end-4))},gradient_y{str2num(d(ii).name(6:end-4))}];
%     % shuffle
%     G(:,ii) = [gradient_x{str2num(d(ii).name(14:end-4))},gradient_y{str2num(d(ii).name(14:end-4))}];
end
N = length(G);
for jj =1:round(N^0.5)
    if mod(N,jj) == 0
        m = jj;
    end
end
index = alpha_estimator(m, G);
% figure
% histogram(G_alpha)
% title(sprintf('mean=%g,std=%g',nanmean(G_alpha),nanstd(G_alpha)))
nanmean(index)
nanstd(index)

% Corollary 2.4 in Mohammadi 2014
function index = alpha_estimator(m, X)
% X is N by d matrix
N = length(X);
n = round(N/m); % must be an integer
Y = sum(reshape(X,n, m,[]),2);

for ii = 1:n
    Y_log_norm(ii) =log(norm(Y(ii,:),'fro') + eps);
end
Y_log_norm_mean = mean(Y_log_norm);
for ii = 1:n
    X_log_norm(ii) = log(norm(X(ii,:),'fro') + eps);
end
X_log_norm_mean = mean(X_log_norm);
diff = (Y_log_norm_mean - X_log_norm_mean) /log(m);
index =  1 / diff;
end