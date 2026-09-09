function cfg = defaultConfig(rootDir)
%DEFAULTCONFIG Central experiment configuration.

if nargin < 1
    rootDir = fileparts(fileparts(mfilename("fullpath")));
end

cfg.seed = 42;

cfg.paths.root = rootDir;
cfg.paths.raw = fullfile(rootDir,"data","raw");
cfg.paths.cache = fullfile(rootDir,"data","cache");
cfg.paths.results = fullfile(rootDir,"results");

cfg.data.mode = "NCLT";
cfg.data.session = "2013-01-10";
cfg.data.autoDownload = true;
cfg.data.downloadHokuyo = true;
cfg.data.keepArchives = true;
cfg.data.dt = 0.20;                 % 5 Hz benchmark timeline
cfg.data.maxSamples = 2500;         % bound runtime; set Inf for full overlap
cfg.data.startOffsetSamples = 1;
cfg.data.gpsMaxAge = 0.35;
cfg.data.hokuyoMaxAge = 0.15;
cfg.data.cachePrepared = true;
cfg.data.forceRebuild = false;

cfg.split.trainFraction = 0.55;
cfg.split.valFraction = 0.15;

cfg.lidar.maxRange = 28.0;
cfg.lidar.minRange = 0.30;
cfg.lidar.beamStride = 6;
cfg.lidar.maxPointsICP = 220;
cfg.lidar.icpIterations = 8;
cfg.lidar.icpMaxCorrespondence = 1.4;
cfg.lidar.icpTrimFraction = 0.72;
cfg.lidar.minICPInliers = 25;

cfg.learning.hiddenDim = 18;
cfg.learning.epochs = 24;
cfg.learning.batchSize = 12;
cfg.learning.learningRate = 2e-3;
cfg.learning.weightDecay = 1e-5;
cfg.learning.trainStride = 2;
cfg.learning.window = 8;            % 1.6 s at 5 Hz
cfg.learning.patience = 5;
cfg.learning.gradientClip = 5.0;

cfg.targets.gpsSigma = 5.0;         % supervision scale, m
cfg.targets.lidarSigma = 2.5;
cfg.targets.odomSigma = 4.0;
cfg.targets.smoothWindow = 5;

cfg.pf.initialN = 800;
cfg.pf.fixedN = 1000;
cfg.pf.minN = 300;
cfg.pf.maxN = 3500;
cfg.pf.baseN = 850;
cfg.pf.failureThreshold = 5.0;

cfg.pf.kldEpsilon = 0.08;
cfg.pf.kldDelta = 0.01;
cfg.pf.kldBinXY = 0.75;
cfg.pf.kldBinYaw = deg2rad(8);

cfg.pf.essResampleRatio = 0.52;
cfg.pf.essBudgetGain = 1250;
cfg.pf.difficultyBudgetGain = 1150;

cfg.pf.lambdaFloor = 0.025;
cfg.pf.broadBase = 0.05;
cfg.pf.broadGain = 0.30;

cfg.pf.motionSigmaXY = 0.16;
cfg.pf.motionSigmaYaw = deg2rad(2.5);
cfg.pf.motionNoiseGain = 1.8;

% Reliability-dependent measurement floors / gains.
cfg.pf.sigmaGpsFloor = 0.90;
cfg.pf.sigmaGpsBase = 0.70;
cfg.pf.sigmaGpsGain = 8.0;

cfg.pf.sigmaLidarFloor = 0.30;
cfg.pf.sigmaLidarBase = 0.18;
cfg.pf.sigmaLidarGain = 2.5;
cfg.pf.sigmaLidarYawFloor = deg2rad(2.0);
cfg.pf.sigmaLidarYawGain = deg2rad(16.0);

cfg.pf.sigmaGpsFixed = 4.5;
cfg.pf.sigmaLidarFixed = 1.4;
cfg.pf.sigmaLidarYawFixed = deg2rad(8.0);

cfg.pf.broadSigmaXY = 3.5;
cfg.pf.broadSigmaYaw = deg2rad(25);

cfg.visual.live = true;
cfg.visual.liveUpdateEvery = 10;
cfg.visual.maxParticleDraw = 700;
cfg.visual.saveFig = true;
end
