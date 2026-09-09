function policy = buildPolicy(name,r,essRatio,kldN,gpsAvailable,lidarAvailable,cfg)
%BUILDPOLICY Map fixed/rule/learned reliability to bounded PF settings.

name = string(name);

switch name
    case "PF-Fixed"
        rG = double(gpsAvailable);
        rL = 0.75*double(lidarAvailable);
        rO = 0.80;
        D = 0.45;
        Ntarget = cfg.pf.fixedN;
        lambda = [1 0 0 0];
        sigG = cfg.pf.sigmaGpsFixed;
        sigL = cfg.pf.sigmaLidarFixed;
        sigLYaw = cfg.pf.sigmaLidarYawFixed;

    case "PF-Adaptive"
        rG = double(gpsAvailable);
        rL = 0.75*double(lidarAvailable);
        rO = 0.80;
        D = 0.45;
        Ntarget = clamp(round(max(kldN, ...
            cfg.pf.baseN + cfg.pf.essBudgetGain*(1-essRatio))),cfg.pf.minN,cfg.pf.maxN);
        lambda = [1 0 0 0];
        sigG = cfg.pf.sigmaGpsFixed;
        sigL = cfg.pf.sigmaLidarFixed;
        sigLYaw = cfg.pf.sigmaLidarYawFixed;

    otherwise
        rG=clamp(r(1),0,1);
        rL=clamp(r(2),0,1);
        rO=clamp(r(3),0,1);
        D=clamp(r(4),0,1);

        raw = [0.22+0.78*rO, rG, rL, cfg.pf.broadBase+cfg.pf.broadGain*D];
        raw(2) = raw(2)*double(gpsAvailable);
        raw(3) = raw(3)*double(lidarAvailable);
        raw = max(raw,cfg.pf.lambdaFloor*[1 double(gpsAvailable) double(lidarAvailable) 1]);
        if ~gpsAvailable, raw(2)=0; end
        if ~lidarAvailable, raw(3)=0; end
        lambda = raw/sum(raw);

        sigG = max(cfg.pf.sigmaGpsFloor, ...
            cfg.pf.sigmaGpsBase + cfg.pf.sigmaGpsGain*(1-rG)^2);
        sigL = max(cfg.pf.sigmaLidarFloor, ...
            cfg.pf.sigmaLidarBase + cfg.pf.sigmaLidarGain*(1-rL)^2);
        sigLYaw = max(cfg.pf.sigmaLidarYawFloor, ...
            cfg.pf.sigmaLidarYawFloor + cfg.pf.sigmaLidarYawGain*(1-rL)^2);

        adaptive = cfg.pf.baseN + cfg.pf.essBudgetGain*(1-essRatio) + ...
                   cfg.pf.difficultyBudgetGain*D;
        Ntarget = clamp(round(max(kldN,adaptive)),cfg.pf.minN,cfg.pf.maxN);
end

processScale = 1 + cfg.pf.motionNoiseGain*(1-rO) + 0.6*D;

policy.r = [rG rL rO D];
policy.lambda = lambda;
policy.Ntarget = Ntarget;
policy.processSigma = [cfg.pf.motionSigmaXY*processScale, ...
                       cfg.pf.motionSigmaXY*processScale, ...
                       cfg.pf.motionSigmaYaw*processScale];
policy.sigmaGps = sigG;
policy.sigmaLidar = sigL;
policy.sigmaLidarYaw = sigLYaw;
end
