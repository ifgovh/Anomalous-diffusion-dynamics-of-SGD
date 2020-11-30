function [MSD,tau] = get_no_average_MSD(Trajectory)

% measures the mean squared displacement (MSD) from a trajectory without time average.

% Trajectory = list of T positions (x,y,t). Or (x,t) or (x,y,z,t) or even
% higher dimension (x1,x2,x3,...t). 

% tau is the list of all the possible time intervals within the trajectory.
% MSD is a list of mean squared displacements, for each value of tau.

% MSD(tau) = x(1)^2 - x(1+tau)^2

increment = Trajectory(:,1:end-1) - Trajectory(:,1);
MSD = sum(increment.^2,2);
tau = 1:size(Trajectory,1);

