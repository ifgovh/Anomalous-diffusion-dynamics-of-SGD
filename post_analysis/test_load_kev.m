clear
close all
%% load data
d = dir('/project/RDS-FSC-THAP-RW/Anomalous-diffusion-dynamics-of-SGD/trained_nets/resnet14_epoch*');
alpha_ls = zeros(1,length(d));
% resnet14_epoch list with MSD, etc
% [3,5,6,7] 
for ii = 1:
    clearvars -except d ii
    close all

datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
select_num = length(datax_dir)-1;
for part = [1,round(select_num/5),select_num]%1:length(datax_dir)-1
    try
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),...
            'MSD','tau','MSD_noaverage','tau_noaverage','F_grads','F_full_batch_grad','F_gradient_noise_norm','Grads','Full_grads','G_noise_norm');
        
        MSD{part} = L.MSD;
        tau{part} = L.tau;
       
        MSD_noaverage{part} = L.MSD_noaverage;
        tau_noaverage{part} = L.tau_noaverage;
        
        F_grads{part} = L.F_grads;
        F_full_batch_grad{part} = L.F_full_batch_grad;
        F_gradient_noise_norm{part} = L.F_gradient_noise_norm;
        Grads{part} = L.Grads;
        Full_grads{part} = L.Full_grads;
        G_noise_norm{part} = L.G_noise_norm;
    catch        
        MSD{part} = nan;
        tau{part} = nan;
        
        MSD_noaverage{part} = nan;
        tau_noaverage{part} = nan;
        
        F_grads{part} = nan;
        F_full_batch_grad{part} = nan;
        F_gradient_noise_norm{part} = nan;
        Grads{part} = nan;
        Full_grads{part} = nan;
        G_noise_norm{part} = nan;
    end

%kkk=1;
% for k = [1,round(select_num/5),select_num]
%     loglog(tau{k},MSD{k},'color',map(kkk,:),'linewidth',linewidth)
%     kkk=kkk+1;
% end
% dist = fitdist(MSD{1},'stable');
% % estimate first regime tail index 
% alpha_ls(ii) = dist.alpha;
end
end