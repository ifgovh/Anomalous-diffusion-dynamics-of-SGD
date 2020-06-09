clear
close all

landscape_type = 'fractal';%'convex'%'random_shuffle'

spatial_epson = 1e-3;
%learning rate
eta = 0.02;
% noise std
eta2 = 0.01;
n = 1000; %number of samples
num_repeat = 5;
[X,Y,gradient_x,gradient_y,pdf_Tr,pdf_G,noise] = deal(cell(num_repeat,1));
land_ind = zeros(num_repeat,1);
for rd = 1:num_repeat
    x_old = rand*2-1; %initial condition
    y_old = rand*2-1; %initial condition
    
    X{rd} = zeros(n+1,1);
    Y{rd} = zeros(n+1,1);
    
    X{rd}(1) = x_old;
    Y{rd}(1) = y_old;
    
    x_new = x_old;
    y_new = y_old;
    
    switch landscape_type
        case 'fractal'
            L = load('../simplified_model/bigger_fractal_landscape.mat');
            x = L.x;
            y = L.y;
            zz = L.zz;
            land_ind(rd) = randperm(length(zz),1);
            z = zz{land_ind(rd)};
        case 'convex'
            load('toy_gaussian_landscape.mat')
        case 'random_shuffle'
            L = load('../simplified_model/bigger_fractal_landscape.mat');
            x = L.x;
            y = L.y;
            zz = L.zz;
            land_ind(rd) = randperm(length(zz),1);
            z = zz{land_ind(rd)};
            z = z(:);
            % shuffle
            z = z(randperm(length(z)));
            z = reshape(z,sqrt(length(z)),sqrt(length(z)));
            % smooth the shuffled landscape to make it differentiable
            z = (imgaussfilt(z,8)).^0.7;
    end
    % interpolate the landscape to get more accurate gradients
    landscape_F = griddedInterpolant((repmat(x',1,length(y))-1)*2-1,(repmat(y,length(x),1)-1)*2-1,z,'linear');% may changet to nonlinear interpolation
    look_up_table = -1:spatial_epson:1;
    [xq, yq] = ndgrid(look_up_table);
    landscape = 10-landscape_F(xq, yq);
    [drift_x,drift_y] = gradient(landscape,spatial_epson);
    % iteration
    for i = 1:n
        x_ind = find(look_up_table > x_old,1);
        y_ind = find(look_up_table > y_old,1);
        
        drift_x_tmp = drift_x(y_ind, x_ind);
        drift_y_tmp = drift_y(y_ind, x_ind);
        
        gradient_y{rd}(i) = drift_y_tmp;
        gradient_x{rd}(i) = drift_y_tmp;
        % make sure the random walker is in th
        counter = 0;
        initial_con_flag = false;
        while true
            %diffusion term
            g = randn;
            x_new = x_old  - drift_x_tmp*eta + eta2*g;
            %diffusion term
            g = randn;
            y_new = y_old  - drift_y_tmp*eta + eta2*g;
            
            counter = counter + 1;
            if abs(x_new) <= 1 && abs(y_new) <= 1
                break
            end
            if counter > 500
                initial_con_flag = true;
                break
            end
        end
        
        if initial_con_flag
            continue
        end
        
        noise{rd}(i) = g;
        
        X{rd}(i+1) = x_new;
        Y{rd}(i+1) = y_new;
        
        x_old = x_new;
        y_old = y_new;
    end
    if initial_con_flag
        continue
    end
    % analysis
    step_increment = [diff(X{rd});diff(Y{rd})];
    step_increment = [step_increment;-step_increment];
    pdf_Tr{rd} = fitdist(step_increment,'stable');
    t = (1:n+1)';
    [MSD{rd},tau{rd}] = get_MSD([X{rd},Y{rd},t]);
    G_pool = [gradient_x{rd},gradient_y{rd}];
    pdf_G{rd} = fitdist(G_pool','stable');
end

save(['../simplified_model/simplified_model_',landscape_type,'.mat'],'X','Y','MSD','tau','pdf_G','pdf_Tr','gradient_y','gradient_x','land_ind','eta','eta2','noise','-v7.3')


function [MSD,tau,displacement_all] = get_MSD(Trajectory,grid_size)
T =  size(Trajectory,1); % T is the number of point in the trajectory;
try
    [ I, j ] = find(triu(ones(T), 1)); % list of indices of possible pairings
catch
    I = [];
    for ii = 1:(T-1)
        temp = 1:ii;
        I = [I;temp'];
    end
    
    j = [];
    for ii = 2:T
        temp = ones(ii-1,1)*ii;
        j = [j;temp];
    end
end

D = zeros(T, T);

if nargin < 2
    % non periodic boundary conditons
    try
        D( I + T*(j-1) ) = (sum(abs( Trajectory(I,1:end-1) - Trajectory(j,1:end-1) ).^2, 2)); % Squared distance computation in one line !
    catch
        % not enough memory
        for ii = 1:length(I)
            D( I(ii) + T*(j(ii)-1) ) = (sum(abs( Trajectory(I(ii),1:end-1) - Trajectory(j(ii),1:end-1) ).^2, 2));
        end
    end
else
    % periodic boundary conditons
    D( I + T*(j-1) ) = (sum(abs( (wrapToPi(Trajectory(I,1:end-1) - Trajectory(j,1:end-1)))./2/pi*grid_size ).^2, 2)); % Squared distance computation in one line !
end

displacement_all = D;

% Time intervals between the two points of each pairing :
dt = zeros(T, T);
dt( I + T*(j-1) ) = -( Trajectory(I,end) - Trajectory(j,end) );

% Then D is a list of squared distances. dt is a list of corresponding
% time intervals. Now we have to average all D values over different dt
% values

% We first remove all 0 values from dt matrix, and from D as well.
idx_0=find(dt==0);
dt(idx_0)=[];
D(idx_0)=[];

% Then we sort dt in ascending order, and sort D in the same way.
[DT,idx]=sort(dt(:));
DD=D(idx);
% We now have DD and DT, respectively a list of squared distances, and
% the corresponding time intervals.

% Now we have to compute the mean DD for each possible value of DT.
% Let's get the first and last indices of each set of DT
First_idx=find(DT-circshift(DT,1)~=0);
Last_idx=find(DT-circshift(DT,-1)~=0);
% For instance, DT=1 start at First_idx(1) and end at Last_idx(1)
%               DT=2 start at First_idx(2) and end at Last_idx(2)...

% To get the average, we first compute the cumulative (starting from 0), then we
% get "the derivative of the cumulative".
C=cumsum([0,DD]);
% For each possible value of DT, the mean of DD is the difference between
% the initial index and the last index of the cumulative, divided by the
% number of elements between those two indices :
MSD=(C(Last_idx+1)-C(First_idx))'./(Last_idx-First_idx+1);
tau=DT(First_idx); % list of intervals
end