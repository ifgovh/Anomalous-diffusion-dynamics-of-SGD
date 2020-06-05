clear
d = dir('*sgd*');
loss_all = cell(length(d),1);
for ii = 1:length(d)
    %GN_dir = dir(fullfile(d(ii).folder,d(ii).name,'*gradient_noise*mat'));
    sub_loss_w_dir = dir(fullfile(d(ii).folder,d(ii).name,'model*.t7'));
    datax_dir = dir(fullfile(sub_loss_w_dir(1).folder,'*data_part*'));
    
    for part = 1:length(datax_dir)
        try
            L = load(fullfile(sub_loss_w_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'delta_train_loss');
%             F = fitdist(L.delta_train_loss,'Stable');
%             alpha(ii,part) = F.alpha;
%             dur{ii,part} = calculate_avalance_dur(L.delta_train_loss);
%             [a(ii,part), xmin(ii,part), Loglike(ii,part)] = plfit(dur{ii,part},'finite');
%             if part == 1 || part == 3 || part == round(0.9*length(datax_dir))
%                 h=plplot(dur{ii,part}, xmin(ii,part), a(ii,part));
%                 close all
%             end
            loss_all{ii} = [loss_all{ii};L.delta_train_loss];
        catch            
        end
        
              
    end
    dur_all{ii} = calculate_avalance_dur(loss_all{ii});
    [a_all(ii), xmin_all(ii), Loglike_all(ii)] = plfit(dur_all{ii},'finite');
    h=plplot(dur_all{ii}, xmin_all(ii), a_all(ii));
    figure
    histogram(loss_all{ii})
    set(gca,'xscale','log','yscale','log')
    [alpha, xmin, L]=plfit(abs(loss_all{ii}));
    plplot(abs(loss_all{ii}), xmin, alpha)
end
pool{1} = 'alex';
pool{2} = 'alex';
pool{3} = 'alex';
pool{4} = 'resnet110noshort';
pool{5} = 'resnet110noshort';
pool{6} = 'resnet110noshort';
pool{7} = 'resnet110';
pool{8} = 'resnet110';
pool{9} = 'resnet56noshort';
pool{10} = 'resnet56noshort';
pool{11} = 'resnet56noshort';
pool{12} = 'resnet56';
pool{13} = 'resnet56';

batch = [1024,128,512,128,256,512,128,256,1024,128,512,1024,512];
figure
for ii=1:15
    subplot(3,5,ii)
    temp= alpha(ii,:);
    temp = temp(temp >0);
    plot(temp)
    title([pool{ii},', mini-batch:',num2str(batch(ii))])
end

subplot(3,5,6)
ylabel('\alpha of {\Delta}L')

for j = 11:15
    subplot(3,5,j)
    xlabel('part')
end

function dur = calculate_avalance_dur(L)
L = abs(L);
SD = std(L);
bool_vec = L > SD;
CC = bwconncomp(bool_vec);
dur = cellfun(@length,CC.PixelIdxList);
end