function [idx,d2] = nearestNeighbor2D(refPts,queryPts)
%NEARESTNEIGHBOR2D Toolbox-free nearest-neighbor search for moderate point sets.

nq = size(queryPts,1);
idx = zeros(nq,1);
d2 = inf(nq,1);

block = 120;
for s = 1:block:nq
    e = min(nq,s+block-1);
    Q = queryPts(s:e,:);
    D2 = (Q(:,1)-refPts(:,1).').^2 + (Q(:,2)-refPts(:,2).').^2;
    [d2(s:e),idx(s:e)] = min(D2,[],2);
end
end
