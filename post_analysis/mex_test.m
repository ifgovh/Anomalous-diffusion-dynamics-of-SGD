clear
aa=10000;
Trajectory = single([rand(aa,500), (1:aa)']);
T =  size(Trajectory,1); % T is the number of point in the trajectory;

[ I, j ] = find(triu(ones(T), 1)); % list of indices of possible pairings
I = uint32(I);
j = uint32(j);
increment = diff(Trajectory(:,1:end-2));
increment2 = [0;sqrt(sum(increment.^2,2,'omitnan'))];

% 
% tic
% [D3,D4] = msd_iteration_mex(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D3,D4] = msd_iteration_mex(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D3,D4] = msd_iteration_mex(Trajectory,increment2, T, I, j, 1);
% toc

% tic
% [D1,D2] = msd_iteration(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D1,D2] = msd_iteration(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D1,D2] = msd_iteration(Trajectory,increment2, T, I, j, 1);
% toc

% tic
% [D5,D6] = msd_iteration_parfor(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D5,D6] = msd_iteration_parfor(Trajectory,increment2, T, I, j, 1);
% toc
% 
% tic
% [D5,D6] = msd_iteration_parfor(Trajectory,increment2, T, I, j, 1);
% toc

tic
[D5,D6] = msd_iteration_parfor_mex(Trajectory,increment2, T, I, j, 1);
toc

tic
[D5,D6] = msd_iteration_parfor_mex(Trajectory,increment2, T, I, j, 1);
toc

tic
[D5,D6] = msd_iteration_parfor_mex(Trajectory,increment2, T, I, j, 1);
toc
