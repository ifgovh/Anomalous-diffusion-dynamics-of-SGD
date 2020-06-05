%generate shuffled fractal landscape
% load('bigger_fractal_landscape.mat')
ss = cellfun(@shuffle_land,zz,'UniformOutput',false);
clear zz
zz = ss;
save('shuffled_landscape.mat','x','y','zz')

function shuff = shuffle_land(z_value)
shuff = z_value(randperm(numel(z_value)));
shuff = reshape(shuff,size(z_value));
% smooth
h = fspecial('average',5);
shuff = imfilter(shuff,h);
end