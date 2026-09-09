function data = prepareNCLT(cfg)
%PREPARENCLT Download, parse, synchronize, and cache one NCLT session.

dtTag = strrep(sprintf("%0.2f",cfg.data.dt),".","p");
if isfinite(cfg.data.maxSamples)
    nTag = sprintf("%d",round(cfg.data.maxSamples));
else
    nTag = "all";
end
% The cached timeline depends on the sample window, so both bounds belong in
% the cache key. Otherwise a short debug run silently poisons a full run.
cacheName = sprintf("NCLT_%s_dt_%s_n_%s_s_%d_hok_%d.mat", ...
    cfg.data.session,dtTag,nTag,round(cfg.data.startOffsetSamples), ...
    double(logical(cfg.data.downloadHokuyo)));
if ~exist(cfg.paths.cache,"dir"), mkdir(cfg.paths.cache); end
cachePath = fullfile(cfg.paths.cache,cacheName);

if cfg.data.cachePrepared && exist(cachePath,"file") && ~cfg.data.forceRebuild
    fprintf("Loading prepared cache: %s\n",cachePath);
    S = load(cachePath,"data");
    data = S.data;
    return;
end

paths = downloadNCLT(cfg);

gt = readmatrix(paths.groundTruth);
if size(gt,2)<7
    error("Unexpected NCLT ground-truth format.");
end
gtTime = gt(:,1)*1e-6;
gtPose = gt(:,2:7);
gtPose2 = [gtPose(:,1:2),gtPose(:,6)];

odomFile = findFileRecursive(paths.extractRoot,"odometry_mu_100hz.csv");
gpsFile = findFileRecursive(paths.extractRoot,"gps.csv");
imuFile = findFileRecursive(paths.extractRoot,"ms25.csv");
odomCovFile = findFileRecursive(paths.extractRoot,"odometry_cov_100hz.csv");
hokFile = findFileRecursive(paths.extractRoot,"hokuyo_30m.bin");

mustExist(odomFile,"odometry_mu_100hz.csv");
mustExist(gpsFile,"gps.csv");

odom = readmatrix(odomFile);
gps = readmatrix(gpsFile);

odomTime = odom(:,1)*1e-6;
odomPoseRaw = [odom(:,2:3),odom(:,7)];

gpsTime = gps(:,1)*1e-6;
gpsFix = gps(:,2);
gpsSV = gps(:,3);
gpsLocal = gpsToLocalNCLT(gps(:,4),gps(:,5),gps(:,6));
gpsSpeed = gps(:,8);

if strlength(imuFile)>0
    imu = readmatrix(imuFile);
    imuTime = imu(:,1)*1e-6;
    imuYawRate = imu(:,10);
    imuAccel = sqrt(sum(imu(:,5:7).^2,2));
else
    imuTime = odomTime;
    imuYawRate = gradient(unwrap(odomPoseRaw(:,3)),odomTime);
    imuAccel = zeros(size(imuYawRate));
end

if strlength(odomCovFile)>0
    oc = readmatrix(odomCovFile);
    ocTime = oc(:,1)*1e-6;
    % upper triangular 6x6: xx is first, yy is 7th upper entry, yaw-yaw last.
    odomCovXY = max(0,oc(:,2)) + max(0,oc(:,8));
    odomYawVar = max(0,oc(:,end));
else
    ocTime = odomTime;
    odomCovXY = 0.2*ones(size(odomTime));
    odomYawVar = deg2rad(3)^2*ones(size(odomTime));
end

t0 = max([gtTime(1),odomTime(1),gpsTime(1)]);
t1 = min([gtTime(end),odomTime(end),gpsTime(end)]);
timeline = (t0:cfg.data.dt:t1).';

if isfinite(cfg.data.maxSamples)
    s = max(1,cfg.data.startOffsetSamples);
    e = min(numel(timeline),s+cfg.data.maxSamples-1);
    timeline = timeline(s:e);
end

gt2 = interpPose(gtTime,gtPose2,timeline);
odom2raw = interpPose(odomTime,odomPoseRaw,timeline);

% Align odometry run-start frame to the NCLT global frame using only initial pose.
Talign = se2Compose(gt2(1,:),se2Inverse(odom2raw(1,:)));
odom2 = zeros(size(odom2raw));
for i=1:size(odom2,1)
    odom2(i,:) = se2Compose(Talign,odom2raw(i,:));
end

[gidx,gAge] = nearestTimeIndex(gpsTime,timeline);
gpsXY = gpsLocal(gidx,1:2);
gpsValid = (gAge<=cfg.data.gpsMaxAge) & gpsFix(gidx)>=2;
gpsXY(~gpsValid,:) = NaN;
gpsFixT = gpsFix(gidx);
gpsSVT = gpsSV(gidx);
gpsSpeedT = gpsSpeed(gidx);

[iidx,~] = nearestTimeIndex(imuTime,timeline);
yawRateT = imuYawRate(iidx);
accelT = imuAccel(iidx);

odomCovXYT = interp1(ocTime,odomCovXY,timeline,"linear","extrap");
odomYawVarT = interp1(ocTime,odomYawVar,timeline,"linear","extrap");

if cfg.data.downloadHokuyo && strlength(hokFile)>0
    fprintf("Reading Hokuyo stream...\n");
    hs = readHokuyo30m(hokFile,cfg);
    [hidx,hAge] = nearestTimeIndex(hs.time,timeline);

    rsel = hs.ranges(:,hidx);
    stale = hAge > cfg.data.hokuyoMaxAge;
    rsel(:,stale) = NaN;

    fprintf("Estimating scan-to-scan Hokuyo odometry (%d synchronized scans)...\n",numel(timeline));
    lidar = estimateLidarOdometry(rsel,hs.angles,odom2,gt2(1,:),cfg);
else
    warning("Hokuyo data unavailable. LiDAR channel falls back to degraded synthetic quality.");
    lidar.pose = odom2;
    lidar.validRatio = 0.05*ones(numel(timeline),1);
    lidar.structure = zeros(numel(timeline),1);
    lidar.icpInlier = zeros(numel(timeline),1);
    lidar.icpRmse = 2*ones(numel(timeline),1);
end

data.time = timeline - timeline(1);
data.utimeSec = timeline;
data.gt = gt2;
data.odom = odom2;
data.gpsXY = gpsXY;
data.gpsValid = gpsValid;
data.gpsFix = gpsFixT;
data.numSV = gpsSVT;
data.gpsSpeed = gpsSpeedT;
data.yawRate = yawRateT;
data.accelNorm = accelT;
data.odomCovXY = odomCovXYT;
data.odomYawVar = odomYawVarT;
data.lidarPose = lidar.pose;
data.lidarValidRatio = lidar.validRatio;
data.lidarStructure = lidar.structure;
data.lidarIcpInlier = lidar.icpInlier;
data.lidarIcpRmse = lidar.icpRmse;
data.source = "NCLT " + string(cfg.data.session);

data = buildSchemaFeatures(data,cfg);

if cfg.data.cachePrepared
    fprintf("Saving prepared cache: %s\n",cachePath);
    save(cachePath,"data","-v7.3");
end
end

function mustExist(pathValue,label)
if strlength(pathValue)==0 || ~exist(pathValue,"file")
    error("Could not locate required NCLT file: %s",label);
end
end
