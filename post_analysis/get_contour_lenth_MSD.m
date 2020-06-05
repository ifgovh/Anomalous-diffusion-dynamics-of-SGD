function [MSD,tau,MSD_forcontour,contour_length,Displacement_all,Contour_length_all] = get_contour_lenth_MSD(Trajectory)

% measures the mean squared displacement (MSD) from a trajectory.
% The code is written by Maxime Deforet, May 21 2013. MSKCC.
% Contact : maxime.deforet@gmail.com

% This code contains no loop. Each step is "vectorized", therefore pretty
% fast.
% The idea is to compute all the possible pairings in only one time.

% Trajectory = list of T positions (x,y,t). Or (x,t) or (x,y,z,t) or even
% higher dimension (x1,x2,x3,...t)
% tau is the list of all the possible time intervals within the trajectory.
% MSD is a list of mean squared displacements, for each value of tau.

% MSD(tau) = sqrt( x(i)^2 - x(i+tau)^2 ), averaged over i.

% contour_length is the contour length (path length) of the trajectory 
% MSD_forcontour is a list of mean squared displacements, for each value of contour_length.

T =  size(Trajectory,1); % T is the number of point in the trajectory;

[ I, j ] = find(triu(ones(T), 1)); % list of indices of possible pairings
% D = zeros(T, T);
% % contour length pool
% D2 = zeros(T, T);

increment = diff(Trajectory(:,1:end-1));
increment2 = [0;sqrt(sum(increment.^2,2,'omitnan'))];

% % non periodic boundary conditons
% try
%     D( I + T*(j-1) ) = (sum(abs( Trajectory(I,1:end-1) - Trajectory(j,1:end-1) ).^2, 2)); % Squared distance computation in one line !
% catch
%     % not enough memory
%     for ii = 1:length(I)
%         D( I(ii) + T*(j(ii)-1) ) = (sum(abs( Trajectory(I(ii),1:end-1) - Trajectory(j(ii),1:end-1) ).^2, 2));
%     end
% end
% 
% for ii = 1:length(I)    
%     D2( I(ii) + T*(j(ii)-1) ) = sum(increment2(min(I(ii),j(ii)):max(I(ii),j(ii))),'omitnan');
% end
[D,D2] = msd_iteration_parfor_mex(Trajectory,increment2, T, I, j, 1);


% Time intervals between the two points of each pairing :
dt = zeros(T, T);
dt( I + T*(j-1) ) = -( Trajectory(I,end) - Trajectory(j,end) );

% Then D is a list of squared distances. dt is a list of corresponding
% time intervals. Now we have to average all D values over different dt
% values

% save a copy for different time moments' displacement
Displacement_all = D;
Contour_length_all = D2;


% We first remove all 0 values from dt matrix, and from D as well.
idx_0=find(dt==0);
dt(idx_0)=[];
D(idx_0)=[];
D2(idx_0)=[];
% Then we sort dt in ascending order, and sort D in the same way.
[DT,idx]=sort(dt(:));
DD=D(idx);

[D2_sort,idx_contour]=sort(D2(:));
DD_forcontour=D(idx_contour);
% We now have DD and DT, respectively a list of squared distances, and
% the corresponding time intervals.

% Now we have to compute the mean DD for each possible value of DT.
% Let's get the first and last indices of each set of DT
First_idx=find(DT-circshift(DT,1)~=0);
Last_idx=find(DT-circshift(DT,-1)~=0);
% For instance, DT=1 start at First_idx(1) and end at Last_idx(1)
%               DT=2 start at First_idx(2) and end at Last_idx(2)...

% same stratege for contour length
First_idx_forcontour=find(D2_sort-circshift(D2_sort,1)~=0);
Last_idx_forcontour=find(D2_sort-circshift(D2_sort,-1)~=0);

% To get the average, we first compute the cumulative (starting from 0), then we
% get "the derivative of the cumulative".
C=cumsum([0,DD],'omitnan');
% For each possible value of DT, the mean of DD is the difference between
% the initial index and the last index of the cumulative, divided by the
% number of elements between those two indices :
MSD=(C(Last_idx+1)-C(First_idx))'./(Last_idx-First_idx+1); 
tau=DT(First_idx); % list of intervals

C_forcontour = cumsum([0,DD_forcontour],'omitnan');
MSD_forcontour = (C_forcontour(Last_idx_forcontour+1)-C_forcontour(First_idx_forcontour))'./(Last_idx_forcontour-First_idx_forcontour+1); 
contour_length=D2_sort(First_idx_forcontour); % list of intervals


% plot(tau,MSD)