clear
h5_name = 'model_300.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5';
loss=h5read(h5_name,'/train_loss');
x=h5read(h5_name,'/xcoordinates');
y=h5read(h5_name,'/ycoordinates');
y = flipud(y);
num_samples = 1000;
[xx,yy,ll] = deal(zeros(num_samples,1));
parfor rd=1:num_samples
    x_ind = randperm(length(x),1);
    y_ind = randperm(length(y),1);
    xx(rd) = x(x_ind);
    yy(rd) = y(y_ind);    
    ll(rd) = loss(y_ind,x_ind);
end
dist = round(pdist([xx,yy]),2);
loss_diff = round(pdist(ll),2);

[C,~,ic] = unique(dist);
[loss_,std_] = deal(zeros(1,length(C)));
parfor kk =1:length(C)
    loss_(kk) = mean(loss_diff(ic == kk),'omitnan');
    std_(kk) = std(loss_diff(ic == kk),'omitnan');
end

save('projected_landscape_fbm.mat','loss_diff','dist','C','loss_','std_')

figure
plot(C,loss_,'.')

xlabel('Distance')
ylabel('Mean squared loss')
set(gca,'xscale','log','yscale','log','fontsize',12)

