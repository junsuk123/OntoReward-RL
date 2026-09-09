function data = buildSchemaFeatures(data,cfg)
%BUILDSCHEMAFEATURES Standardized runtime features and GT-only supervision targets.

T = numel(data.time);

gpsInnovation = nan(T,1);
gpsSpeedResidual = nan(T,1);

for t=1:T
    if data.gpsValid(t)
        gpsInnovation(t) = norm(data.gpsXY(t,:)-data.odom(t,1:2));
        if t>1
            odomSpeed = norm(data.odom(t,1:2)-data.odom(t-1,1:2))/max(eps,data.time(t)-data.time(t-1));
        else
            odomSpeed = data.gpsSpeed(t);
        end
        gpsSpeedResidual(t) = abs(data.gpsSpeed(t)-odomSpeed);
    end
end

gpsInnovation = fillmissing(gpsInnovation,"constant",25);
gpsSpeedResidual = fillmissing(gpsSpeedResidual,"constant",5);

dynamics = clamp(0.65*abs(data.yawRate)/0.8 + 0.35*abs(data.accelNorm-9.81)/3.0,0,2);

openSkyScore = 0.40*double(data.gpsFix>=2) + ...
    0.30*clamp(data.numSV/10,0,1) + ...
    0.30*exp(-gpsInnovation/8);
openSkyScore(~data.gpsValid) = 0;

% Runtime feature table: no GT-derived errors are included.
data.features = table( ...
    double(data.gpsFix(:)), double(data.numSV(:)), gpsInnovation(:), gpsSpeedResidual(:), ...
    double(data.lidarValidRatio(:)), double(data.lidarIcpInlier(:)), double(data.lidarIcpRmse(:)), ...
    double(data.lidarStructure(:)), double(data.odomCovXY(:)), double(data.odomYawVar(:)), ...
    dynamics(:), openSkyScore(:), ...
    'VariableNames',["GPSFix","NumSV","GPSInnovation","GPSSpeedResidual", ...
    "LidarValidRatio","LidarInlierRatio","LidarICPRMSE","LidarStructure", ...
    "OdomCovXY","OdomYawVar","Dynamics","OpenSkyScore"]);

% GT is used only to create training/evaluation targets.
gpsErr = 30*ones(T,1);
ok = data.gpsValid & all(isfinite(data.gpsXY),2);
gpsErr(ok) = sqrt(sum((data.gpsXY(ok,:)-data.gt(ok,1:2)).^2,2));

lidarErr = sqrt(sum((data.lidarPose(:,1:2)-data.gt(:,1:2)).^2,2));
odomErr = sqrt(sum((data.odom(:,1:2)-data.gt(:,1:2)).^2,2));

rG = double(ok).*exp(-0.5*(gpsErr/cfg.targets.gpsSigma).^2);
qL = clamp(0.35*data.lidarValidRatio + 0.40*data.lidarIcpInlier + ...
    0.25*exp(-data.lidarIcpRmse/0.8),0,1);
rL = qL .* exp(-0.5*(lidarErr/cfg.targets.lidarSigma).^2);

odomQ = exp(-0.8*sqrt(max(0,data.odomCovXY))) .* exp(-0.25*dynamics);
rO = odomQ .* exp(-0.5*(odomErr/cfg.targets.odomSigma).^2);

rG = clamp(rG,0,1);
rL = clamp(rL,0,1);
rO = clamp(rO,0,1);

difficulty = 1 - (0.55*max([rG,rL,rO],[],2) + 0.45*mean([rG,rL,rO],2));
difficulty = clamp(difficulty + 0.10*clamp(dynamics,0,1),0,1);

w = cfg.targets.smoothWindow;
if w>1
    rG = movmean(rG,w);
    rL = movmean(rL,w);
    rO = movmean(rO,w);
    difficulty = movmean(difficulty,w);
end

data.targets = table(rG,rL,rO,difficulty,gpsErr,lidarErr,odomErr, ...
    'VariableNames',["GNSSReliability","LiDARReliability","OdomReliability", ...
    "Difficulty","GPSError","LidarError","OdomError"]);
end
