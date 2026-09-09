function cfg = defaultConfig(mode)
%DEFAULTCONFIG Central configuration for physics, ontology, R-GAT and PPO.
% mode: 'quick' (smoke/iteration) or 'full' (paper-scale Monte Carlo)
if nargin < 1, mode = 'full'; end
cfg.mode = lower(mode);
cfg.seed = 42;

%% Simulation
cfg.sim.dt = 0.02;              % s (50 Hz rigid-body integration)
cfg.sim.maxTime = 14.0;         % s
cfg.sim.maxSteps = round(cfg.sim.maxTime/cfg.sim.dt);
cfg.sim.integrator = 'RK4';
cfg.sim.g = 9.80665;
cfg.sim.realtime = false;
cfg.sim.monitorEvery = 5;
cfg.sim.worldXYLimit = 12.0;
cfg.sim.maxAltitude = 12.0;
cfg.sim.crashTilt = deg2rad(75);
cfg.sim.groundZ = 0.06;

%% SJTU-like quadrotor default rigid body
% Mass/inertia are aligned with a commonly circulated SJTU drone model.
% Geometry below is an explicit approximation and should be replaced by CAD/URDF values.
cfg.drone.mass = 1.477; % kg
cfg.drone.J = diag([0.1152, 0.1152, 0.218]); % kg m^2
cfg.drone.bodyDims = [0.34, 0.34, 0.10]; % x,y,z [m], approximate
cfg.drone.armLength = 0.23;     % m from CG to rotor
cfg.drone.armWidth  = 0.035;
cfg.drone.armHeight = 0.025;
cfg.drone.rotorRadius = 0.10;   % m
cfg.drone.maxTotalThrust = 30.0; % N, consistent with sjtu_drone plugin limit

%% Propulsion / motor dynamics
cfg.prop.kT = 1.80e-5;          % N/(rad/s)^2, calibrated for ~450 rad/s hover
cfg.prop.kQ = 2.8e-7;           % N m/(rad/s)^2
cfg.prop.omegaMin = 0;
cfg.prop.omegaMax = sqrt(cfg.drone.maxTotalThrust/(4*cfg.prop.kT));
cfg.prop.tauMotor = 0.045;      % s first-order motor speed response
cfg.prop.spinDir = [1;-1;1;-1]; % reaction torque signs
L = cfg.drone.armLength/sqrt(2);
cfg.prop.rotorPosB = [ L, -L, -L,  L;
                       L,  L, -L, -L;
                       0,  0,  0,  0];
cfg.prop.groundEffect.enable = true;
cfg.prop.groundEffect.maxGain = 1.14;
cfg.prop.groundEffect.heightScale = 2.2*cfg.drone.rotorRadius;

%% Atmosphere / distributed aerodynamics
cfg.aero.rho0 = 1.225;          % kg/m^3 sea-level reference
cfg.aero.T0 = 288.15;           % K
cfg.aero.p0 = 101325;           % Pa
cfg.aero.lapse = 0.0065;        % K/m
cfg.aero.Rair = 287.05;         % J/(kg K)
cfg.aero.mu0 = 1.7894e-5;       % Pa s reference
cfg.aero.Suth = 110.4;          % K
cfg.aero.CdNormal = 1.20;       % flat/bluff panel pressure-drag coefficient
cfg.aero.panelGrid = 2;         % per fuselage face dimension; increase to 3-4 for full study
cfg.aero.armPanelGrid = 1;
cfg.aero.includeArms = true;
cfg.aero.minRe = 1.0;
cfg.aero.transitionRe = 5e5;
cfg.aero.panels = aero.makeSurfacePanels(cfg);

%% Spatially varying wind field (not a single-direction gust)
cfg.wind.meanRef = [1.3; 0.4; 0.0]; % m/s at zRef
cfg.wind.zRef = 10.0;
cfg.wind.z0 = 0.25;
cfg.wind.shearAlpha = 0.20;
cfg.wind.verticalMean = 0.0;
cfg.wind.turbulenceIntensity = 0.55; % m/s RMS-scale
cfg.wind.numModes = 10;
cfg.wind.vortex.enable = true;
cfg.wind.vortex.center = [0.5;-0.5;2.0];
cfg.wind.vortex.coreRadius = 0.8;
cfg.wind.vortex.strength = 1.2; % m^2/s

% Deterministic Fourier turbulence modes for exact reproducibility.
s = RandStream('mt19937ar','Seed',cfg.seed+7);
M = cfg.wind.numModes;
K = randn(s,3,M);
K = K ./ max(vecnorm(K),1e-9);
scales = linspace(0.8,5.0,M); % rad/m spatial frequencies
cfg.wind.modes.K = K .* scales;
D = randn(s,3,M);
D = D ./ max(vecnorm(D),1e-9);
cfg.wind.modes.D = D;
cfg.wind.modes.amp = cfg.wind.turbulenceIntensity*(0.35+0.65*rand(s,1,M))/sqrt(M/2);
cfg.wind.modes.omega = 0.3 + 2.2*rand(s,1,M); % rad/s
cfg.wind.modes.phase = 2*pi*rand(s,1,M);

% Localized gust packets. Each acts with a different spatial vector field.
g(1) = struct('center',[0.0;0.0;2.5],'t0',4.0,'sigmaXYZ',[1.8;1.0;1.2], ...
              'sigmaT',0.8,'vector',[2.1;-0.9;0.5]);
g(2) = struct('center',[0.8;-0.4;1.2],'t0',7.2,'sigmaXYZ',[0.9;1.6;0.8], ...
              'sigmaT',0.55,'vector',[-1.4;1.8;-0.7]);
g(3) = struct('center',[-0.6;0.7;0.55],'t0',10.0,'sigmaXYZ',[1.1;0.8;0.45], ...
              'sigmaT',0.45,'vector',[0.8;-1.1;1.0]);
cfg.wind.gusts = g;

%% Sensor / vision abstraction
cfg.sensor.posStd = 0.015;       % m
cfg.sensor.velStd = 0.025;       % m/s
cfg.sensor.angleStd = deg2rad(0.25);
cfg.sensor.rateStd = deg2rad(0.8);
cfg.sensor.marker.maxRange = 7.0;
cfg.sensor.marker.baseDetect = 0.995;
cfg.sensor.marker.tiltScale = deg2rad(32);
cfg.sensor.marker.windScale = 0.75;
cfg.sensor.marker.xyScale = 2.0;

%% Semantic features derived from the user's landing-paper formulation
cfg.semantic.windAccelThr = 4.0;         % m/s^2
cfg.semantic.windDirThr = deg2rad(50);   % rad
cfg.semantic.windRiskW = [2.0;1.4;1.0];  % configurable assumption
cfg.semantic.windRiskB = -1.4;
cfg.semantic.alignScale = 0.75;
cfg.semantic.attTiltScale = deg2rad(22);
cfg.semantic.attRateScale = deg2rad(80);
cfg.semantic.vzSafeScale = 0.65;

%% Landing safety criteria (evaluation ground truth, NOT training reward weights)
cfg.criteria.xy = 0.28;                  % m
cfg.criteria.vz = 0.55;                  % m/s magnitude at contact
cfg.criteria.tilt = deg2rad(10);
cfg.criteria.rate = deg2rad(45);

%% RL observation/action
cfg.rl.obsDim = 15;
cfg.rl.actDim = 4; % collective, roll command, pitch command, yaw-rate command
cfg.rl.maxRollPitch = deg2rad(28);
cfg.rl.maxYawRate = deg2rad(90);
cfg.rl.collectiveSpan = 0.85; % hover*(1 + span*a1)
cfg.rl.KpAtt = [4.8;4.8;2.1];
cfg.rl.KdAtt = [1.5;1.5;0.9];

%% Manual dense baseline reward (deliberately hand-designed)
cfg.reward.manual.wPos = 1.3;
cfg.reward.manual.wVel = 0.35;
cfg.reward.manual.wTilt = 0.65;
cfg.reward.manual.wRate = 0.10;
cfg.reward.manual.wWind = 0.45;
cfg.reward.manual.wAct = 0.025;
cfg.reward.manual.success = 20;
cfg.reward.manual.failure = -20;
cfg.reward.manual.timeout = -20;   % never touching down is a mission failure
cfg.reward.manual.time = 0.25;     % per-second pressure so hovering is not free
cfg.reward.manual.violSpan = 2.0;  % criteria-ratio span over which the crash penalty grades in
cfg.reward.manual.failureFloor = 0.25; % fraction of the crash penalty applied to a near miss
cfg.reward.sparse.success = 10;
cfg.reward.sparse.failure = -10;
cfg.reward.sparse.time = -0.15;
cfg.reward.sparse.timeout = -10;
cfg.reward.pbrs.lambda = 2.0;
cfg.reward.pbrs.gamma = 0.999;   % must match cfg.ppo.gamma for PBRS invariance

%% Ontology graph / R-GAT
cfg.ontology.nodeNames = {'PositionError','VerticalSpeed','TiltAngle','AngularRate', ...
    'WindRisk','MarkerQuality','VisualStability','Alignment','AttitudeStability', ...
    'TouchdownSafety','SafeLanding'};
cfg.ontology.relationNames = {'degrades','supports','contributes','self'};
cfg.ontology.nNodes = numel(cfg.ontology.nodeNames);
cfg.ontology.nRelations = numel(cfg.ontology.relationNames);
cfg.ontology.inDim = 4 + cfg.ontology.nNodes;
cfg.rgat.hiddenDim = 24;
cfg.rgat.relDim = 6;
cfg.rgat.lr = 2e-3;
cfg.rgat.batchSize = 32;
cfg.rgat.sampleStride = 3;
% Behavior-policy perturbation is sampled log-uniformly: the expert success
% boundary sits near sigma=0.05, so a uniform sweep to 0.65 would label ~96%%
% of the dataset negative and leave the potential with nothing to fit.
cfg.rgat.noiseRange = [0.02 0.65];

%% PPO (custom implementation; Deep Learning Toolbox only)
cfg.ppo.hidden = 64;
cfg.ppo.gamma = 0.999;        % horizon ~1000 steps > maxSteps, so terminal rewards are visible
cfg.ppo.lambdaGAE = 0.95;
cfg.ppo.clip = 0.20;
cfg.ppo.entropyCoef = 0.003;
cfg.ppo.valueCoef = 0.5;
cfg.ppo.actorLR = 2e-4;
cfg.ppo.criticLR = 7e-4;
cfg.ppo.epochs = 5;
cfg.ppo.minibatch = 64;
cfg.ppo.rolloutSteps = 2048;  % batch many episodes before each update
cfg.ppo.initLogStd = -0.55;
cfg.ppo.gradClip = 5.0;

switch cfg.mode
    case 'quick'
        cfg.rgat.dataEpisodes = 24;
        cfg.rgat.epochs = 10;
        cfg.ppo.trainEpisodes = 150;
        cfg.eval.episodes = 12;
        cfg.eval.sweepEpisodes = 6;
        cfg.aero.panelGrid = 2;
    case 'full'
        cfg.rgat.dataEpisodes = 400;
        cfg.rgat.epochs = 80;
        cfg.ppo.trainEpisodes = 3000;
        cfg.eval.episodes = 200;
        cfg.eval.sweepEpisodes = 50;
        cfg.aero.panelGrid = 3;
        cfg.aero.panels = aero.makeSurfacePanels(cfg);
    otherwise
        error('Unknown mode: %s', mode);
end

cfg.eval.seed0 = 5000;
cfg.eval.windScales = [0.5, 1.0, 1.5, 2.0];
cfg.viz.training = true;
cfg.viz.realtime = true;
cfg.viz.liveEvery = 5;      % episodes (or epochs) between live snapshot exports
cfg.viz.liveExport = true;  % also write PNG + CSV so progress is visible headless
cfg.paths.root = fileparts(fileparts(mfilename('fullpath')));
cfg.paths.results = fullfile(cfg.paths.root,'results');
cfg.paths.models = fullfile(cfg.paths.results,'models');
cfg.paths.figures = fullfile(cfg.paths.results,'figures');
cfg.paths.data = fullfile(cfg.paths.results,'data');
cfg.paths.live = fullfile(cfg.paths.results,'live');
for d = {cfg.paths.results,cfg.paths.models,cfg.paths.figures,cfg.paths.data,cfg.paths.live}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end
end
