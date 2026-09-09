function [delta,quality] = icp2d(prevPts,currPts,initDelta,cfg)
%ICP2D Lightweight robust point-to-point 2-D ICP.
% Finds T_prev_curr mapping current-scan points into previous-scan frame.

delta = initDelta(:).';
quality = struct("rmse",Inf,"inlierRatio",0,"numInliers",0);

if size(prevPts,1) < cfg.lidar.minICPInliers || size(currPts,1) < cfg.lidar.minICPInliers
    return;
end

for iter = 1:cfg.lidar.icpIterations
    q = transformPointsSE2(currPts,delta);

    [nnIdx,d2] = nearestNeighbor2D(prevPts,q);
    d = sqrt(d2);
    valid = d < cfg.lidar.icpMaxCorrespondence;

    if nnz(valid) < cfg.lidar.minICPInliers
        break;
    end

    dv = d(valid);
    [~,ord] = sort(dv,"ascend");
    nkeep = max(cfg.lidar.minICPInliers,round(cfg.lidar.icpTrimFraction*numel(ord)));
    nkeep = min(nkeep,numel(ord));

    vv = find(valid);
    keep = vv(ord(1:nkeep));

    A = q(keep,:);
    B = prevPts(nnIdx(keep),:);

    ca = mean(A,1);
    cb = mean(B,1);
    AA = A-ca;
    BB = B-cb;

    H = AA.'*BB;
    [U,~,V] = svd(H);
    R = V*U.';
    if det(R)<0
        V(:,2) = -V(:,2);
        R = V*U.';
    end
    t = cb.' - R*ca.';
    dtheta = atan2(R(2,1),R(1,1));

    inc = [t(1),t(2),dtheta];
    delta = se2Compose(inc,delta);

    if norm(inc(1:2)) < 1e-4 && abs(inc(3)) < 1e-4
        break;
    end
end

q = transformPointsSE2(currPts,delta);
[nnIdx,d2] = nearestNeighbor2D(prevPts,q); %#ok<ASGLU>
d = sqrt(d2);
valid = d < cfg.lidar.icpMaxCorrespondence;

quality.numInliers = nnz(valid);
quality.inlierRatio = nnz(valid)/max(1,size(currPts,1));
if any(valid)
    quality.rmse = sqrt(mean(d2(valid)));
end
end
