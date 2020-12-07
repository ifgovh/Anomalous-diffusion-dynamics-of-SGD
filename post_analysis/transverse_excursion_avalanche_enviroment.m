function transverse_excursion_avalanche_enviroment(varargin)
% load data
d = dir('*net*');
[end_end_dis2,transversed_excursion,delta_loss_all,loss_all] = deal(cell(length(d),1));
fragment_length = 50;
% Loop number for PBS array job
loop_num = 0;
for ii = 1:length(d)
    % For PBS array job
    loop_num = loop_num + 1;
    if nargin ~= 0
        PBS_ARRAYID = varargin{1};
        if loop_num ~=  PBS_ARRAYID
            continue;
        end
    end
    
    datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
    [end_end_dis2{ii},transversed_excursion{ii}] = deal(cell(length(datax_dir),1));
    for part = 1:length(datax_dir)
        L = load(fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']),'MSD_feed','delta_train_loss','loss');
        % transversed excursion, randomly get 100 fragments
        for rd = 1:100
            fragment_start = randperm(size(L.MSD_feed,1) - fragment_length,1);
            fragment_end = fragment_start + fragment_length;
            temp_tran_excu = zeros(fragment_length,1);
            end_end_dis2{ii}{part}(rd) = sum((L.MSD_feed(fragment_start,1:end-2) - L.MSD_feed(fragment_end,1:end-2)).^2);
            for jj = 1:fragment_length
                CA = L.MSD_feed(fragment_start + jj,1:end-2) - L.MSD_feed(fragment_start,1:end-2);
                BA = L.MSD_feed(fragment_end,1:end-2) - L.MSD_feed(fragment_start,1:end-2);
                % distance from point C to line AB
                temp_tran_excu(jj) = norm(CA-dot(CA,BA)/dot(BA,BA)*BA);
            end
            transversed_excursion{ii}{part}(rd) = max(temp_tran_excu)^2;
        end
        %         figure
        %         plot(end_end_dis2{ii}{part},transversed_excursion{ii}{part},'ko')
        %         xlabel('{\Delta}R^2')
        %         ylabel('Transverse excursion^2')
        %         set(gca,'xscale','log','yscale','log')
        %         drawnow
        % get loss
        delta_loss_all{ii} = [delta_loss_all{ii};L.delta_train_loss];
        loss_all{ii} = [loss_all{ii};L.loss];
        % around avalanche, loss vs contour length, delta loss vs contour length
        [seg_loss{ii}{part},seg_delta_loss{ii}{part},contour_len{ii}{part}] = get_avalance_enviro(L.delta_train_loss,L.loss,L.MSD_feed(:,1:end-2),0);
    end
    %     [seg_loss_total{ii},seg_delta_loss_total{ii},contour_len_total{ii}] = get_avalance_enviro(delta_loss_all{ii},loss_all{ii},L.MSD_feed(:,1:end-2),0);
    save(['trans_ex_avalan_',num2str(ii),'.mat'],'transversed_excursion','seg_loss','seg_delta_loss','contour_len','loss_all','delta_loss_all','end_end_dis2','-v7.3')
end
end
function [seg_loss,seg_delta_loss,contour_len] = get_avalance_enviro(delta_L,L,X,plot_flag)
delta_L = abs(delta_L);
SD = std(delta_L);
bool_vec = delta_L > SD;
CC = bwconncomp(bool_vec(1:end-50));
len_aval = cellfun(@length,CC.PixelIdxList);
[~,I] = maxk(len_aval,5);
for ii = 1:5
    [~,ind] = max(delta_L(CC.PixelIdxList{I(ii)}));
    index(ii) = CC.PixelIdxList{I(ii)}(ind);
    seg_loss(:,ii) = L(index(ii):index(ii)+50);
    seg_delta_loss(:,ii) = delta_L(index(ii):index(ii)+50);
    % calculate contour length
    seg = X(index(ii):index(ii)+50,:);
    contour_len(:,ii) = [0;cumsum(sqrt(sum(diff(seg,1,1).^2,2)))];
end
if plot_flag
    figure
    subplot(121)
    hold on
    for jj = 1:5%randperm(length(CC.PixelIdxList),5)
        plot(contour_len(:,jj),seg_loss(:,jj))
    end
    
    subplot(122)
    hold on
    for jj = 1:5%randperm(length(CC.PixelIdxList),5)
        plot(contour_len(:,jj),seg_delta_loss(:,jj))
    end
    drawnow
end
end