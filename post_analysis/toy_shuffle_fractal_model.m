clear
close all

spatial_epson = 1e-3;
%learning rate
eta = 0.02;%x5
% noise portion
eta2 = 0.01;%x5
n = 1000; %number of samples
num_rd = 5;
[X,Y,gradient_x,gradient_y,pdf_Tr,pdf_G,noise] = deal(cell(num_rd,1));
land_ind = zeros(num_rd,1);
for rd = 1:num_rd
    x_old = rand*2-1; %initial condition
    y_old = rand*2-1; %initial condition
    
    X{rd} = zeros(n+1,1);
    Y{rd} = zeros(n+1,1);
    
    X{rd}(1) = x_old;
    Y{rd}(1) = y_old;
    
    x_new = x_old;
    y_new = y_old;
    
    % generate fractal landscape
    % [x,y,z] = fractal_landscape_generator();% x,y in [1,2]
    L = load('bigger_fractal_landscape.mat');
    x = L.x;
    y = L.y;
    zz = L.zz;
    land_ind(rd) = randperm(length(zz),1);
    z = zz{land_ind(rd)};
    % load('toy_fractal_landscape.mat')
    % load('toy_gaussian_landscape.mat')
    % x0 = -1:spatial_epson:1;
    % [x,y]=ndgrid(x0);
    % landscape = 10-10*exp(-(x.^2 + y.^2)./2);
    z = z(:);
    z = z(randperm(length(z)));
    z = reshape(z,sqrt(length(z)),sqrt(length(z)));
    % smooth the shuffled landscape
    z = (imgaussfilt(z,8)).^0.7;
    landscape_F = griddedInterpolant((repmat(x',1,length(y))-1)*2-1,(repmat(y,length(x),1)-1)*2-1,z,'linear');% may changet to nonlinear interpolation
    look_up_table = -1:spatial_epson:1;
    [xq, yq] = ndgrid(look_up_table);
    landscape = 10-landscape_F(xq, yq);
    landscape_pool{rd} = landscape;
    [drift_x,drift_y] = gradient(landscape,spatial_epson);
%     imagesc(drift_x);colorbar
    alpha = 2;
    for i = 1:n
        %     if  mod(10*i/n, 1) == 0
        %         progress = i/n
        %     end
        %drift term
        x_ind = find(look_up_table > x_old,1);
        y_ind = find(look_up_table > y_old,1);
        
        drift_x_tmp = drift_x(y_ind, x_ind);
        drift_y_tmp = drift_y(y_ind, x_ind);
        
        gradient_y{rd}(i) = drift_y_tmp;
        gradient_x{rd}(i) = drift_y_tmp;
        while true
            %diffusion term
            %g = stblrnd(alpha,0,0.9,0);%randn;
            g = randn;
            x_new = x_old  - drift_x_tmp*eta + eta2*g;
            
            %diffusion term
            %g = stblrnd(alpha,0,0.9,0);%randn;
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
    
    figure('visible','on')
    subplot(221)
    step_increment = [diff(X{rd});diff(Y{rd})];
    step_increment = [step_increment;-step_increment];
    histogram(step_increment)
    pdf_Tr{rd} = fitdist(step_increment,'stable');
    
    subplot(222)
    t = (1:n+1)';
    [MSD,tau] = get_MSD([X{rd},Y{rd},t]);
    loglog(tau,MSD)
    xlabel('\tau')
    ylabel('MSD')
    
    subplot(223)
    G_pool = [gradient_x{rd},gradient_y{rd}];
    histogram(G_pool)
    pdf_G{rd} = fitdist(G_pool','stable');
    % set(gca,'xscale','log','yscale','log')
    
    subplot(224)
    hold on
    contour(look_up_table,look_up_table,landscape)
    % step = 1;
    % color_map = jet(round(length(X)/2+1));
    % for ii=1:step:(length(X)/2+1)
    %     plot(X(ii),Y(ii),'.','color',color_map(ii,:))
    % end
    plot(X{rd},Y{rd})
    plot(X{rd}(end),Y{rd}(end),'ro')
    plot(X{rd}(1),Y{rd}(1),'rx')
    xlabel('X')
    ylabel('Y')
    
    saveas(gcf,sprintf('test_shuffle_%d.jpg',rd))
end
save('test_toy_shuffle_fractal.mat','X','Y','pdf_G','pdf_Tr','gradient_y','gradient_x','landscape_pool','eta','eta2','noise','-v7.3')