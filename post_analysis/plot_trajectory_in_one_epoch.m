function plot_trajectory_in_one_epoch(w,model_folder,part_num)
%Plot the optimization path in the space spanned by principle directions.
args.model_folder = model_folder; %'folders for models to be projected')
% args.start_epoch = start_epoch;%'min index of epochs')
% args.max_epoch = max_epoch; %'max number of epochs')
args.part_num = part_num;
args.save_epoch = 1; %save models every few epochs')

my_setup_PCA_directions(args, w)
end

function my_setup_PCA_directions(args, w)
% Name the .h5 file that stores the PCA directions.
folder_name = [args.model_folder, '/PCA_tiny_epoch', num2str(args.part_num)];

eval(['mkdir ',folder_name])
dir_name = [folder_name, '/directions_matlab.h5'];

% skip if the direction file exists
% dir_skip = dir(dir_name);
% if isempty(dir_skip)

% prepare the optimization path matrix

matrix = w - w(end,:);

% Perform PCA on the optimization path matrix
disp("Perform PCA on the models")
[coeff,score,latent,~,explained,~] = pca(matrix);

pc1 = score(:,1);
pc2 = score(:,2);
fprintf("angle between pc1 and pc2: %g", cal_angle(pc1, pc2))

fprintf("pca.explained_variance_ratio, pc 1: %g, pc 2: %g", explained(1),explained(2))

% save to hdf5 and let python code do the interface part
hdf5write(dir_name,'/explained_variance_',latent(1:2))
hdf5write(dir_name,'/explained_variance_ratio_',explained(1:2),'WriteMode','append');
hdf5write(dir_name,'/directionx',coeff(:,1),'WriteMode','append');
hdf5write(dir_name,'/directiony',coeff(:,2),'WriteMode','append');

proj_name = [folder_name, '/directions.h5_proj_cos.h5'];
hdf5write(proj_name,'/proj_xcoord',score(:,1))
hdf5write(proj_name,'/proj_ycoord',score(:,2),'WriteMode','append')

end

function cosine_sim = cal_angle(vec1, vec2)
% Calculate cosine similarities between two torch tensors or two ndarraies
%    Args:
%        vec1, vec2: two tensors
cosine_sim = dot(vec1, vec2)/ (norm(vec1)*norm(vec2));
end





