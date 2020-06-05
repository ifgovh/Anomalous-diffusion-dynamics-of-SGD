% % use DNN loss landscape
% xcoordinates = h5read('/import/headnode1/gche4213/Project3/plot_data/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,201]x[-1.0,1.0,201].h5','/xcoordinates');
% ycoordinates = h5read('/import/headnode1/gche4213/Project3/plot_data/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,201]x[-1.0,1.0,201].h5','/ycoordinates');
% train_loss = h5read('/import/headnode1/gche4213/Project3/plot_data/model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,201]x[-1.0,1.0,201].h5','/train_loss');
% 
% surface(xcoordinates,ycoordinates,log(train_loss))
% view(3)
% shading interp
% xlabel('w1')
% ylabel('w2')
% zlabel('Loss')

% use a fractal landscape
% R = load('/import/headnode1/gche4213/Project3/post_analysis/bigger_fractal_landscape.mat');
figure
surface(R.x,R.y,R.zz{1})
colormap(flip(brewermap([],'Spectral')))
view(3)
shading interp
material shiny%metal
% lighting gouraud
xlabel('w1')
ylabel('w2')
zlabel('Loss')