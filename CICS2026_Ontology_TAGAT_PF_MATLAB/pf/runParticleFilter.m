function result = runParticleFilter(data,testIdx,name,rel,cfg,runDir)
%RUNPARTICLEFILTER Mixture-proposal sequential Monte Carlo localization.

K = numel(testIdx);
idx0 = testIdx(1);

N = cfg.pf.initialN;
init = data.gt(idx0,:) + [0.4*randn,0.4*randn,deg2rad(4)*randn];
particles = [ ...
    init(1)+0.7*randn(N,1), ...
    init(2)+0.7*randn(N,1), ...
    wrapAngle(init(3)+deg2rad(8)*randn(N,1))];
weights = ones(N,1)/N;

est = nan(K,3);
Psave = nan(3,3,K);
ESS = nan(K,1);
Nlog = nan(K,1);
runtimeMs = nan(K,1);
rlog = nan(K,4);
lamlog = nan(K,4);
siglog = nan(K,3);

live = [];
if cfg.visual.live
    live = initLiveVisualizer(data,testIdx,name,cfg);
end

kldN = N;

for kk=1:K
    ticStep = tic;
    t = testIdx(kk);

    if kk==1
        dOdom = [0 0 0];
    else
        dOdom = se2Between(data.odom(testIdx(kk-1),:),data.odom(t,:));
    end

    essRatio = effectiveSampleSize(weights)/numel(weights);

    if isempty(rel)
        rr = [1 1 1 0.5];
    else
        rr = rel.Y(:,t).';
    end

    gpsAvail = data.gpsValid(t) && all(isfinite(data.gpsXY(t,:)));

    % LiDAR scan matching is dead reckoning: the integrated pose drifts without
    % bound (hundreds of metres over an NCLT session), so it cannot be used as
    % an absolute position measurement. Anchor its scan-to-scan delta on the
    % previous posterior mean instead, which is how a scan-matching constraint
    % is meant to enter the filter.
    lidarMeas = [NaN NaN NaN];
    if kk>1
        tPrev = testIdx(kk-1);
        if all(isfinite(data.lidarPose(tPrev,:))) && all(isfinite(data.lidarPose(t,:)))
            dLidar = se2Between(data.lidarPose(tPrev,:),data.lidarPose(t,:));
            lidarMeas = se2Compose(est(kk-1,:),dLidar);
        end
    end

    lidarAvail = all(isfinite(lidarMeas)) && ...
                 data.lidarIcpInlier(t)>0.03 && isfinite(data.lidarIcpRmse(t));

    policy = buildPolicy(name,rr,essRatio,kldN,gpsAvail,lidarAvail,cfg);

    % Ancestors are selected according to previous posterior weights.
    anc = systematicSampleIndices(weights,policy.Ntarget);
    pa = particles(anc,:);

    muMotion = propagatePoseBatch(pa,dOdom);

    [particles,comp] = sampleMixtureProposal(muMotion, ...
        data.gpsXY(t,:),lidarMeas,policy,cfg); %#ok<ASGLU>

    % Importance correction p(x_t|x_{t-1}) / q(x_t|...)
    logPtrans = diagGaussianLogPdfSE2(particles,muMotion,policy.processSigma);
    logQ = mixtureProposalLogPdf(particles,muMotion,data.gpsXY(t,:), ...
        lidarMeas,policy,cfg);

    logLik = zeros(size(logQ));

    if gpsAvail
        logLik = logLik + gaussianLogPdf2D(particles(:,1:2),data.gpsXY(t,:),policy.sigmaGps);
    end

    if lidarAvail
        sig = [policy.sigmaLidar policy.sigmaLidar policy.sigmaLidarYaw];
        logLik = logLik + diagGaussianLogPdfSE2(particles, ...
            repmat(lidarMeas,size(particles,1),1),sig);
    end

    lw = logLik + logPtrans - logQ;
    lw = lw - max(lw);
    weights = exp(lw);
    sw = sum(weights);
    if ~isfinite(sw) || sw<=0
        weights = ones(size(weights))/numel(weights);
    else
        weights = weights/sw;
    end

    [est(kk,:),Psave(:,:,kk)] = weightedPoseStats(particles,weights);
    ESS(kk) = effectiveSampleSize(weights);
    Nlog(kk) = numel(weights);

    kldN = kldRequiredN(particles,cfg);

    if ESS(kk) < cfg.pf.essResampleRatio*numel(weights)
        ids = systematicSampleIndices(weights,numel(weights));
        particles = particles(ids,:);
        weights = ones(numel(ids),1)/numel(ids);
        % mild roughening prevents deterministic collapse.
        particles(:,1:2) = particles(:,1:2) + 0.02*randn(size(particles,1),2);
        particles(:,3) = wrapAngle(particles(:,3)+deg2rad(0.2)*randn(size(particles,1),1));
    end

    runtimeMs(kk) = 1000*toc(ticStep);
    rlog(kk,:) = policy.r;
    lamlog(kk,:) = policy.lambda;
    siglog(kk,:) = [policy.sigmaGps,policy.sigmaLidar,policy.sigmaLidarYaw];

    if cfg.visual.live && (mod(kk,cfg.visual.liveUpdateEvery)==0 || kk==K)
        live = updateLiveVisualizer(live,data,testIdx,kk,est,particles,weights, ...
            ESS,Nlog,rlog,lamlog,name,cfg);
    end
end

tsec = data.time(testIdx)-data.time(testIdx(1));
result.name = string(name);
result.est = est;
result.P = Psave;
result.ESS = ESS;
result.N = Nlog;
result.runtimeMs = runtimeMs;
result.reliability = rlog;
result.lambda = lamlog;

result.trajectory = table(tsec(:),data.gt(testIdx,1),data.gt(testIdx,2),data.gt(testIdx,3), ...
    est(:,1),est(:,2),est(:,3), ...
    'VariableNames',["Time","GT_X","GT_Y","GT_Yaw","Est_X","Est_Y","Est_Yaw"]);

result.stepLog = table(tsec(:),ESS,Nlog,runtimeMs, ...
    rlog(:,1),rlog(:,2),rlog(:,3),rlog(:,4), ...
    lamlog(:,1),lamlog(:,2),lamlog(:,3),lamlog(:,4), ...
    siglog(:,1),siglog(:,2),siglog(:,3), ...
    'VariableNames',["Time","ESS","Particles","Runtime_ms", ...
    "rGNSS","rLiDAR","rOdom","Difficulty", ...
    "lambdaMotion","lambdaGNSS","lambdaLiDAR","lambdaBroad", ...
    "sigmaGNSS","sigmaLiDAR","sigmaLiDARYaw"]);

if cfg.visual.live && ~isempty(live) && isfield(live,"fig") && isvalid(live.fig)
    modelDir = fullfile(runDir,name);
    if ~exist(modelDir,"dir"),mkdir(modelDir);end
    exportgraphics(live.fig,fullfile(modelDir,"live_final.png"),"Resolution",150);
end
end
