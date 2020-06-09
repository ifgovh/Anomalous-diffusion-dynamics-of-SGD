clear
close all

spatial_epson = 1e-3;
%learning rate
eta = 0.02;
% noise std
eta2 = 0.01;
n = 1000; %number of samples
num_repeat = 5;
[X,Y,gradient_x,gradient_y,pdf_Tr,pdf_G,noise] = deal(cell(num_rd,1));
land_ind = zeros(num_rd,1);
for rd = 1:num_rd
    x_old = -1;%rand*2-1; %initial condition
    y_old = -1;%rand*2-1; %initial condition
    
    X{rd} = zeros(n+1,1);
    Y{rd} = zeros(n+1,1);
    
    X{rd}(1) = x_old;
    Y{rd}(1) = y_old;
    
    x_new = x_old;
    y_new = y_old;
    
    switch landscape_type
        case 'fractal'
            L = load('bigger_fractal_landscape.mat');
            x = L.x;
            y = L.y;
            zz = L.zz;
            land_ind(rd) = randperm(length(zz),1);
            z = zz{land_ind(rd)};
        case 'convex'
            load('toy_gaussian_landscape.mat')
        case 'random_shuffle'
            L = load('bigger_fractal_landscape.mat');
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
        while true
            %diffusion term
            g = randn;
            x_new = x_old  - drift_x_tmp*eta + eta2*g;
            %diffusion term
            g = randn;
            y_new = y_old  - drift_y_tmp*eta + eta2*g;
            if abs(x_new) <= 1 && abs(y_new) <= 1
                break
            end
        end
        noise{rd}(i) = g;
        
        X{rd}(i+1) = x_new;
        Y{rd}(i+1) = y_new;
        
        x_old = x_new;
        y_old = y_new;
    end
    % analysis
    step_increment = [diff(X{rd});diff(Y{rd})];
    step_increment = [step_increment;-step_increment];
    pdf_Tr{rd} = fitdist(step_increment,'stable');
    t = (1:n+1)';
    [MSD,tau] = get_MSD([X{rd},Y{rd},t]);
    G_pool = [gradient_x{rd},gradient_y{rd}];
    pdf_G{rd} = fitdist(G_pool','stable');
end
save(['../simplified_model/simplified_model_',landscape_type,'.mat'],'X','Y','pdf_G','pdf_Tr','gradient_y','gradient_x','land_ind','eta','eta2','noise','-v7.3')