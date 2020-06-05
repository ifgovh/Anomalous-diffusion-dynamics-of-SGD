clear
close all
d = dir('resnet*sgd*wd*mom*');
hold on
for ii= 1:length(d)
    eval(['cd ',d(ii).name])
%     % gradient noise
%     d_mat = dir('*gradient_noise.mat');
%     load(d_mat.name, 'train_noise_norm')
%     a=cell2mat(train_noise_norm(:,1));
%     b = a(1:200,:);
%     b = b(:);
%     F{ii}=fitdist(double(b),'stable');
%     histogram(b,'facealpha',0.3,'edgealpha',0.3,'normalization','pdf')  
%     waitforbuttonpress
    d_mat = dir('*all_weights.mat');
    load(d_mat.name, 'weight')
    weight = cellfun(@(x) reshape(x,[],1) ,weight,'UniformOutput',false);
    a=cell2mat(weight');
    grad{ii} = diff(a');
%     F{ii}=fitdist(double(grad{ii}(:)),'stable');
    [N,edges] = histcounts(grad{ii},'normalization','pdf');  
    plot(edges(2:end),N,'.-')
    cd ..
end

save('gradient_dist.mat','grad','d','-v7.3')
saveas(gcf,'gradient_dist.fig')

