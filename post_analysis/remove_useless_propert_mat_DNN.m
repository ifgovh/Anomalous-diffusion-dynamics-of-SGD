% plot figures of Proj3
clear
close all

d = dir('/headnode1/gche4213/Project3/*net*');
for ii = 1:length(d)
    datax_dir = dir(fullfile(d(ii).folder,d(ii).name,'*data_part*'));
    
    for part = 1:length(datax_dir)-1
        clearvars -except d ii datax_dir part
        filename = fullfile(datax_dir(1).folder,[d(ii).name(1:end-24),'_data_part_',num2str(part),'.mat']);
        load(filename)
        eval(['delete ',filename])
        save(filename,'loss','test_loss','delta_train_loss','delta_test_loss',...
            'MSD','Mean_contourlength','tau','MSD_forcontour','MSL_contourlength','contour_length','MSL_distance',...
            'distance','Displacement_all','Contour_length_all','f_loss','P1_loss','f_test_loss','P1_test_loss','-v7.3');
    end
end