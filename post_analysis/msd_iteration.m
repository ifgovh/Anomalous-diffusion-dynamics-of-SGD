function [D,D2] = msd_iteration(Trajectory,increment2, T, I, j, shift)
D = single(zeros(T));
for ii = 1:length(I)
    D( I(ii) + T*(j(ii)-1) ) = (sum(abs( Trajectory(I(ii),1:end-shift) - Trajectory(j(ii),1:end-shift) ).^2, 2));
end

D2 = single(zeros(T));
for ii = 1:length(I)    
    D2( I(ii) + T*(j(ii)-1) ) = sum(increment2(min(I(ii),j(ii)):max(I(ii),j(ii))),'omitnan');
end
end