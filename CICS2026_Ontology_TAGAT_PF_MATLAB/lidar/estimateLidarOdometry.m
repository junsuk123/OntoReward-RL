function lidar = estimateLidarOdometry(ranges,angles,odomPose,initialPose,cfg)
%ESTIMATELIDARODOMETRY Scan-to-scan Hokuyo odometry and quality features.

T = size(ranges,2);
pose = nan(T,3);
pose(1,:) = initialPose;

validRatio = zeros(T,1);
structure = zeros(T,1);
icpInlier = zeros(T,1);
icpRmse = inf(T,1);
delta = zeros(T,3);

prevPts = rangesToPoints(ranges(:,1),angles,cfg);
[validRatio(1),structure(1)] = scanQuality(ranges(:,1),angles,cfg);

for t = 2:T
    currPts = rangesToPoints(ranges(:,t),angles,cfg);
    [validRatio(t),structure(t)] = scanQuality(ranges(:,t),angles,cfg);

    initDelta = se2Between(odomPose(t-1,:),odomPose(t,:));
    [d,q] = icp2d(prevPts,currPts,initDelta,cfg);

    if ~isfinite(q.rmse) || q.numInliers < cfg.lidar.minICPInliers
        d = initDelta;
        q.rmse = cfg.lidar.icpMaxCorrespondence;
        q.inlierRatio = 0;
    end

    delta(t,:) = d;
    pose(t,:) = se2Compose(pose(t-1,:),d);
    icpInlier(t) = q.inlierRatio;
    icpRmse(t) = q.rmse;

    prevPts = currPts;
end

% Seed step 1 from the first successful match; indexing with an empty find
% result would otherwise error when no scan pair matched at all.
firstFinite = find(isfinite(icpRmse),1,"first");
if isempty(firstFinite)
    icpRmse(1) = 1;
else
    icpRmse(1) = icpRmse(firstFinite);
end
icpInlier(1) = icpInlier(min(2,T));

lidar.pose = pose;
lidar.delta = delta;
lidar.validRatio = validRatio;
lidar.structure = structure;
lidar.icpInlier = icpInlier;
lidar.icpRmse = icpRmse;
end

function [validRatio,structure] = scanQuality(ranges,angles,cfg)
r = double(ranges);
valid = isfinite(r) & r>=cfg.lidar.minRange & r<=cfg.lidar.maxRange;
validRatio = mean(valid);

if nnz(valid) < 20
    structure = 0;
    return;
end

rv = r(valid);
av = double(angles(valid));
p = [rv.*cos(av),rv.*sin(av)];
C = cov(p);
ev = sort(eig(C),"descend");
if numel(ev)<2 || sum(ev)<=eps
    structure = 0;
else
    isotropy = min(ev)/max(ev);
    spread = min(1,sqrt(sum(ev))/12);
    structure = min(1,0.55*isotropy/0.35 + 0.45*spread);
end
end
