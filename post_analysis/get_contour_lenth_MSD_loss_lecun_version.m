function [MSD,contour_length,dL] = get_contour_lenth_MSD_loss_lecun_version(Trajectory)

% measures the mean squared displacement (MSD) from a trajectory.
% The code is written by Guozhang Chen, Oct 14. USYD.

% This code contains no loop. Each step is "vectorized", therefore pretty
% fast.
% The idea is to compute all the possible pairings in only one time.

% Trajectory = list of T positions (x,y,t,L). Or (x,t,L) or (x,y,z,t,L) or even
% higher dimension (x1,x2,x3,...t,L). L is the loss.
% tau is the list of all the possible time intervals within the trajectory.
% MSD is a list of mean squared displacements, for each value of tau.

% MSD(tau) = sqrt( x(i)^2 - x(i+tau)^2 ), averaged over i.

% contour_length is the contour length (path length) of the trajectory 
% MSD_forcontour is a list of mean squared displacements, for each value of contour_length.

T =  size(Trajectory,1); % T is the number of point in the trajectory;

step_increment = diff(Trajectory(:,1:end-2));
step_increment2 = [0;sqrt(sum(step_increment.^2,2,'omitnan'))];

diff_Traj = Trajectory(:,1:end-2) - Trajectory(1,1:end-2);

MSD = single(zeros(T,1));
for ii = 1:T
    MSD( ii ) = norm(diff_Traj(ii,:)).^2;
end

contour_length = cumsum(step_increment2);

% Loss difference between the two points of each pairing :
dL = Trajectory(:,end) - Trajectory(1,end);



