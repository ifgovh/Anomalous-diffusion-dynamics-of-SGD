function [D,D2] = msd_iteration_parfor(Trajectory,increment2, T, I, j, shift)
D = single(zeros(T));
temp = single(zeros(numel(D),1));
parfor ii = 1:length(I)
    temp( ii ) = (sum(abs( Trajectory(I(ii),1:end-shift) - Trajectory(j(ii),1:end-shift) ).^2, 2));
end
for ii = 1:length(I)
    D( I(ii) + T*(j(ii)-1) ) = temp(ii);
end

D2 = single(zeros(T));
temp = single(zeros(numel(D),1));

parfor ii = 1:length(I)    
    temp( ii ) = sum(increment2(min(I(ii),j(ii)):max(I(ii),j(ii))),'omitnan');
end

for ii = 1:length(I)    
    D2( I(ii) + T*(j(ii)-1) ) = temp(ii);
end
end