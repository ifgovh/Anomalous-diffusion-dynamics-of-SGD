d = dir('*net*');
for ii = 1:length(d)
    %GN_dir = dir(fullfile(d(ii).folder,d(ii).name,'*gradient_noise*mat'));
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*.t7'));
    datax_dir = dir(fullfile(sub_loss_w_dir(1).folder,'*data_part*'));
    
    for part = 1:length(datax_dir)
        L = load(fullfile(sub_loss_w_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'delta_train_loss');
        F = fitdist(L.delta_train_loss,'Stable');
        alpha(ii,part) = F.alpha;
    end
end
pool{1} = 'resnet110';
pool{2} = 'resnet14';
pool{3} = 'resnet20noshort';
pool{4} = 'resnet20';
pool{5} = 'resnet56';
batch = [512,1024,128,128,128];
figure
for ii=1:5
    subplot(3,2,ii)
    temp= alpha(ii,:);
    temp = temp(temp >0);
    plot(temp)
    title([pool{ii},', mini-batch:',num2str(batch(ii))])
end

subplot(3,2,3)
ylabel('\alpha of {\Delta}L')

subplot(3,2,5)
xlabel('part')

subplot(3,2,4)
xlabel('part')