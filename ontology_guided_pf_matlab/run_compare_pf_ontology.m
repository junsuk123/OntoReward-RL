%% Schema/Ontology-Guided Adaptive Particle Filter demo
% Baseline bootstrap PF vs. proposed ontology-guided adaptive PF, with
%   (a) fair (identical) resampling thresholds for every variant,
%   (b) a 4-stage ablation study  PF-Base / PF-R / PF-RG / PF-Full,
%   (c) Monte-Carlo repetition (filter-seed only, and dataset+filter seed),
%   (d) 3-D visualization of the environment / sensor context, and
%   (e) explicit visualization of how the schema + ontology layer is applied.
%
% No special toolbox required (MATLAB R2021a+ recommended).
%
% The default CSV is a synthetic UrbanNav-like urban localization sequence.
% It contains OpenSky -> UrbanCanyon -> Tunnel -> UrbanCanyon -> FeaturePoor
% -> OpenSky transitions and GNSS/LiDAR quality features.
%
% Authoring purpose: reproducible proof-of-concept for
% "Schema/Ontology-Guided Adaptive Particle Sampling for Reliable UGV Localization"

clear; clc; close all;

rootDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(rootDir, 'data');
resultDir = fullfile(rootDir, 'results');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(resultDir, 'dir'), mkdir(resultDir); end

cfg = defaultConfig();
csvFile = fullfile(dataDir, 'synthetic_urbannav_like.csv');

if cfg.regenerateDataset || ~exist(csvFile, 'file')
    fprintf('[1/7] Generating UrbanNav-like synthetic dataset...\n');
    rng(cfg.datasetSeed, 'twister');
    T = generateUrbanNavLikeDataset(cfg);
    writetable(T, csvFile);
else
    fprintf('[1/7] Loading dataset: %s\n', csvFile);
    T = readtable(csvFile, 'TextType', 'string');
    T = normalizeDatasetTypes(T);
end

%% ---- Single deterministic run of every ablation variant ----------------
V = pfVariants(cfg);
fprintf('[2/7] Running %d PF variants (single run, filterSeed=%d)...\n', numel(V), cfg.filterSeed);
fprintf('       resampling threshold: ');
if cfg.fairResampling
    fprintf('ESS/N < %.2f for ALL variants (fair comparison)\n', cfg.resampleESS);
else
    fprintf('baseline %.2f / proposed %.2f (legacy, NOT fair)\n', cfg.baseResampleESS, cfg.propResampleESS);
end

runs = struct('name', {}, 'label', {}, 'out', {});
for i = 1:numel(V)
    fprintf('       - %-8s %s\n', V(i).name, V(i).label);
    rng(cfg.filterSeed, 'twister');
    runs(i).name = V(i).name;
    runs(i).label = V(i).label;
    runs(i).out = runPF(T, cfg, V(i).opt);
end
base = runs(1).out;   % PF-Base   (kept for backward compatibility)
prop = runs(end).out; % PF-Full

%% ---- Metrics ----------------------------------------------------------
fprintf('[3/7] Computing metrics and ablation table...\n');
[summaryTbl, contextTbl] = computeMetrics(T, runs);
ablationTbl = computeAblationTable(T, runs);

%% ---- Monte Carlo ------------------------------------------------------
mc = struct();
if cfg.runMonteCarlo
    fprintf('[4/7] Monte Carlo: %d runs x %d variants, mode A (filter seed only)...\n', ...
        cfg.mcRuns, numel(V));
    mc.filterSeed = monteCarloStudy(cfg, "filterSeed", T);
    fprintf('       Monte Carlo: %d runs x %d variants, mode B (dataset + filter seed)...\n', ...
        cfg.mcRuns, numel(V));
    mc.datasetSeed = monteCarloStudy(cfg, "datasetAndFilterSeed", T);
else
    fprintf('[4/7] Monte Carlo skipped (cfg.runMonteCarlo = false).\n');
end

%% ---- Save ------------------------------------------------------------
fprintf('[5/7] Writing result files...\n');
writetable(summaryTbl, fullfile(resultDir, 'summary_metrics.csv'));
writetable(contextTbl, fullfile(resultDir, 'context_metrics.csv'));
writetable(ablationTbl, fullfile(resultDir, 'ablation_metrics.csv'));
writetable(ontologyTraceTable(T, prop), fullfile(resultDir, 'ontology_trace.csv'));
if cfg.runMonteCarlo
    writetable(mc.filterSeed.summary,  fullfile(resultDir, 'monte_carlo_filterseed.csv'));
    writetable(mc.filterSeed.context,  fullfile(resultDir, 'monte_carlo_filterseed_context.csv'));
    writetable(mc.datasetSeed.summary, fullfile(resultDir, 'monte_carlo_datasetseed.csv'));
    writetable(mc.datasetSeed.context, fullfile(resultDir, 'monte_carlo_datasetseed_context.csv'));
end
save(fullfile(resultDir, 'pf_comparison_results.mat'), ...
    'T', 'base', 'prop', 'runs', 'cfg', 'summaryTbl', 'contextTbl', 'ablationTbl', 'mc');

%% ---- Report ----------------------------------------------------------
disp(' ');
disp('==================== Overall comparison ====================');
disp(summaryTbl);
disp('==================== Context-wise RMSE =====================');
disp(contextTbl);
disp('==================== Ablation (single run) ==================');
disp(ablationTbl);
if cfg.runMonteCarlo
    fprintf('======= Monte Carlo A: fixed dataset, %d filter seeds =======\n', cfg.mcRuns);
    disp(mc.filterSeed.summary);
    printPairedImprovement(mc.filterSeed);
    fprintf('======= Monte Carlo B: %d dataset + filter seeds ============\n', cfg.mcRuns);
    disp(mc.datasetSeed.summary);
    printPairedImprovement(mc.datasetSeed);
end

%% ---- Plots ----------------------------------------------------------
fprintf('[6/7] Plotting results...\n');
plotComparison(T, runs, contextTbl, resultDir);
plotDiagnostics(T, runs, contextTbl, resultDir);
plotEnvironment3D(T, runs, cfg, resultDir);
plotOntologyPipeline(T, prop, cfg, resultDir);
plotOntologyApplied(T, runs, cfg, resultDir);
plotAblation(T, runs, ablationTbl, contextTbl, mc, cfg, resultDir);
if cfg.runMonteCarlo
    plotMonteCarlo(mc, cfg, resultDir);
end

fprintf('[7/7] Done. Results saved to:\n  %s\n', resultDir);
fprintf('Main outputs:\n');
fprintf('  - summary_metrics.csv / context_metrics.csv / ablation_metrics.csv\n');
fprintf('  - ontology_trace.csv                (per-step schema -> ontology -> PF parameters)\n');
if cfg.runMonteCarlo
    fprintf('  - monte_carlo_filterseed.csv / monte_carlo_datasetseed.csv (+ *_context.csv)\n');
end
fprintf('  - trajectory_and_efficiency.png     (trajectory / error / budget / reliability)\n');
fprintf('  - diagnostics.png                   (ESS / context RMSE / step time / error pdf)\n');
fprintf('  - environment_3d.png                (3-D environment + sensor context)\n');
fprintf('  - ontology_pipeline.png             (schema -> ontology -> PF parameter mapping)\n');
fprintf('  - ontology_applied.png              (what the ontology actually did, per step)\n');
fprintf('  - ablation.png');
if cfg.runMonteCarlo, fprintf(' / monte_carlo.png'); end
fprintf('\n  - pf_comparison_results.mat\n');

%% ========================================================================
%  CONFIGURATION
%% ========================================================================
function cfg = defaultConfig()
    cfg.datasetSeed = 42;
    cfg.filterSeed = 123;
    cfg.regenerateDataset = false;

    % Filter rate: 10 Hz. This can be interpreted as a 50-Hz IMU stream
    % preintegrated/downsampled to a 10-Hz localization update.
    cfg.dt = 0.1;
    cfg.duration_s = 300;

    % Baseline PF
    cfg.Nbase = 2000;

    % Proposed PF adaptive budget
    cfg.Nmin = 600;
    cfg.Nmax = 2000;
    cfg.particleChangeThreshold = 0.15;
    cfg.rougheningPos = 0.05;             % m,   applied only when N grows
    cfg.rougheningYaw = deg2rad(0.15);    % rad, applied only when N grows

    % State [x, y, yaw, v] transition covariance per 0.1 s step.
    cfg.qPos = 0.10;                % m
    cfg.qYaw = deg2rad(0.55);       % rad
    cfg.qVel = 0.10;                % m/s

    % Initial uncertainty shared by both methods.
    cfg.initPosStd = 3.0;           % m
    cfg.initYawStd = deg2rad(8.0);  % rad
    cfg.initVelStd = 0.4;           % m/s

    % Baseline fixed measurement noise assumptions.
    cfg.baseGnssSigma = 1.5;               % m
    cfg.baseLidarPosSigma = 0.70;          % m
    cfg.baseLidarYawSigma = deg2rad(2.0);  % rad

    % ---- Reliability -> measurement covariance mapping ------------------
    % sigma = max(floor, base + gain*(1-r)^2).
    % The floor stops the ontology from becoming over-confident when the
    % schema features look perfect. Without it, OpenSky/Tunnel accuracy
    % degrades because the filter over-trusts the raw measurement.
    cfg.gnssSigmaFloor = 0.90;        cfg.gnssSigmaBase = 0.70;      cfg.gnssSigmaGain = 8.0;
    cfg.lidarPosSigmaFloor = 0.30;    cfg.lidarPosSigmaBase = 0.18;  cfg.lidarPosSigmaGain = 2.5;
    cfg.lidarYawSigmaFloor = deg2rad(0.8);
    cfg.lidarYawSigmaBase = deg2rad(0.8);
    cfg.lidarYawSigmaGain = deg2rad(9.0);
    cfg.reliabilityGate = 0.02;     % below this a source is treated as absent

    % ---- Fair comparison -----------------------------------------------
    % Every variant resamples at the same ESS/N threshold so that any ESS
    % or RMSE difference cannot be attributed to a more aggressive
    % resampling schedule. Set fairResampling = false to reproduce the
    % older asymmetric setting (0.50 baseline vs 0.55 proposed).
    cfg.fairResampling = true;
    cfg.resampleESS = 0.50;
    cfg.baseResampleESS = 0.50;     % legacy, used only if ~fairResampling
    cfg.propResampleESS = 0.55;     % legacy, used only if ~fairResampling

    % ---- Monte Carlo ---------------------------------------------------
    % A single PF run is not a valid result: resampling and proposal
    % sampling are stochastic. mcRuns >= 30 is recommended. Runtime is
    % reported as a MEDIAN over runs, because a single tic/toc in MATLAB
    % is dominated by JIT / CPU-scheduling variation.
    cfg.runMonteCarlo = true;
    cfg.mcRuns = 30;
end

%% ========================================================================
function V = pfVariants(cfg)
% Ablation ladder. Each row adds exactly one proposed component.
%
%   Variant   ontology reliability | guided proposal | adaptive N
%   PF-Base            -                    -              -
%   PF-R               O                    -              -
%   PF-RG              O                    O              -
%   PF-Full            O                    O              O

    if cfg.fairResampling
        essBase = cfg.resampleESS;
        essProp = cfg.resampleESS;
    else
        essBase = cfg.baseResampleESS;
        essProp = cfg.propResampleESS;
    end

    mk = @(rel, guided, adaptN, N, ess) struct( ...
        'useReliability', rel, 'useGuidedProposal', guided, ...
        'useAdaptiveN', adaptN, 'N', N, 'resampleESS', ess);

    V(1).name  = "PF-Base";
    V(1).label = 'bootstrap PF, fixed R, fixed N = 2000';
    V(1).opt   = mk(false, false, false, cfg.Nbase, essBase);

    V(2).name  = "PF-R";
    V(2).label = '+ ontology reliability -> adaptive R';
    V(2).opt   = mk(true,  false, false, cfg.Nbase, essProp);

    V(3).name  = "PF-RG";
    V(3).label = '+ measurement-informed guided proposal';
    V(3).opt   = mk(true,  true,  false, cfg.Nbase, essProp);

    V(4).name  = "PF-Full";
    V(4).label = '+ context/ESS-adaptive particle budget (proposed)';
    V(4).opt   = mk(true,  true,  true,  cfg.Nmax, essProp);
end

%% ========================================================================
%  SYNTHETIC DATASET
%% ========================================================================
function T = generateUrbanNavLikeDataset(cfg)
% Synthetic dataset inspired by the sensor rates and failure modes of an
% urban multisensor localization dataset: GNSS 5 Hz, LiDAR 10 Hz, 10-Hz
% truth/filter timeline, with urban-canyon multipath and tunnel outage.

    dt = cfg.dt;
    t = (0:dt:cfg.duration_s)';
    K = numel(t);

    true_x = zeros(K,1);
    true_y = zeros(K,1);
    true_yaw = zeros(K,1);
    true_v = zeros(K,1);
    true_yaw_rate = zeros(K,1);

    true_v(1) = 3.0;
    for k = 2:K
        tk = t(k);
        vCmd = 3.0 + 0.55*sin(0.028*tk) + 0.25*sin(0.11*tk);
        yawRateCmd = commandedYawRate(tk);
        true_yaw_rate(k) = yawRateCmd;

        true_v(k) = true_v(k-1) + 0.9*(vCmd - true_v(k-1))*dt;
        true_yaw(k) = wrapPiLocal(true_yaw(k-1) + yawRateCmd*dt);
        true_x(k) = true_x(k-1) + true_v(k)*cos(true_yaw(k))*dt;
        true_y(k) = true_y(k-1) + true_v(k)*sin(true_yaw(k))*dt;
    end

    % Controls available to both filters.
    speedBias = filter(1, [1 -0.995], 0.002*randn(K,1));
    gyroBias = filter(1, [1 -0.998], deg2rad(0.002)*randn(K,1));
    speed_meas = true_v + speedBias + 0.06*randn(K,1);
    yaw_rate_meas = true_yaw_rate + gyroBias + deg2rad(0.30)*randn(K,1);

    context = strings(K,1);
    gnss_available = false(K,1);
    gnss_x = nan(K,1); gnss_y = nan(K,1);
    numSV = zeros(K,1); hdop = nan(K,1); cn0_mean = nan(K,1); gnss_hAcc = nan(K,1);

    lidar_available = false(K,1);
    lidar_x = nan(K,1); lidar_y = nan(K,1); lidar_yaw = nan(K,1);
    lidar_score = nan(K,1); lidar_feature_count = zeros(K,1);

    gnssBias = [0 0];
    lidarBias = [0 0];

    for k = 1:K
        c = contextFromTime(t(k));
        context(k) = c;

        % ---------------- GNSS schema features + measurement ----------------
        % GNSS nominal update at 5 Hz on the 10-Hz timeline.
        gnssEpoch = mod(k-1, 2) == 0;
        [sv, d, cno, hacc, pAvail, sigmaG, biasSigma, nlosProb] = gnssContextModel(c);
        numSV(k) = max(0, round(sv + 1.2*randn));
        hdop(k) = max(0.5, d * exp(0.10*randn));
        cn0_mean(k) = max(0, cno + 1.5*randn);
        gnss_hAcc(k) = max(0.2, hacc * exp(0.15*randn));

        if c == "Tunnel"
            gnssBias = 0.995*gnssBias;
        else
            gnssBias = 0.97*gnssBias + biasSigma*randn(1,2);
        end

        if gnssEpoch && rand < pAvail
            gnss_available(k) = true;
            jump = [0 0];
            if rand < nlosProb
                phi = 2*pi*rand;
                mag = 6 + 7*rand;
                jump = mag*[cos(phi), sin(phi)];
                % NLOS/outlier is accompanied by degraded receiver quality
                % indicators, allowing the schema/ontology layer to detect it.
                numSV(k) = max(2, numSV(k)-2);
                hdop(k) = 1.8*hdop(k);
                cn0_mean(k) = max(15,cn0_mean(k)-8);
                gnss_hAcc(k) = 2.0*gnss_hAcc(k);
            end
            g = [true_x(k), true_y(k)] + gnssBias + jump + sigmaG*randn(1,2);
            gnss_x(k) = g(1); gnss_y(k) = g(2);
        end

        % ---------------- LiDAR schema features + pseudo-map-match pose -----
        [pLAvail, sigmaL, sigmaYaw, scoreMean, featMean, outlierProb] = lidarContextModel(c);
        lidar_score(k) = min(1, max(0, scoreMean + 0.06*randn));
        lidar_feature_count(k) = max(0, round(featMean + 35*randn));

        lidarBias = 0.94*lidarBias + 0.05*sigmaL*randn(1,2);
        if rand < pLAvail
            lidar_available(k) = true;
            outlier = [0 0];
            yawOutlier = 0;
            if rand < outlierProb
                phi = 2*pi*rand;
                mag = 2 + 4*rand;
                outlier = mag*[cos(phi), sin(phi)];
                yawOutlier = deg2rad(4 + 5*rand) * sign(randn);
                % Registration outliers typically have poorer matching scores.
                lidar_score(k) = 0.35*lidar_score(k);
                lidar_feature_count(k) = round(0.45*lidar_feature_count(k));
            end
            l = [true_x(k), true_y(k)] + lidarBias + outlier + sigmaL*randn(1,2);
            lidar_x(k) = l(1); lidar_y(k) = l(2);
            lidar_yaw(k) = wrapPiLocal(true_yaw(k) + yawOutlier + sigmaYaw*randn);
        end
    end

    T = table(t, context, true_x, true_y, true_yaw, true_v, ...
        speed_meas, yaw_rate_meas, ...
        gnss_available, gnss_x, gnss_y, numSV, hdop, cn0_mean, gnss_hAcc, ...
        lidar_available, lidar_x, lidar_y, lidar_yaw, lidar_score, lidar_feature_count, ...
        'VariableNames', {'time_s','context','true_x','true_y','true_yaw','true_v', ...
        'speed_meas','yaw_rate_meas','gnss_available','gnss_x','gnss_y','numSV','hdop', ...
        'cn0_mean','gnss_hAcc','lidar_available','lidar_x','lidar_y','lidar_yaw', ...
        'lidar_score','lidar_feature_count'});
end

function w = commandedYawRate(t)
    w = 0;
    if t >= 25 && t < 55
        w = deg2rad(1.25);
    elseif t >= 80 && t < 110
        w = deg2rad(-1.55);
    elseif t >= 132 && t < 158
        w = deg2rad(1.7);
    elseif t >= 185 && t < 215
        w = deg2rad(-1.35);
    elseif t >= 238 && t < 270
        w = deg2rad(1.15);
    end
end

function c = contextFromTime(t)
    if t < 60
        c = "OpenSky";
    elseif t < 120
        c = "UrbanCanyon";
    elseif t < 165
        c = "Tunnel";
    elseif t < 220
        c = "UrbanCanyon";
    elseif t < 255
        c = "FeaturePoor";
    else
        c = "OpenSky";
    end
end

function [sv, hdop, cno, hacc, pAvail, sigmaG, biasSigma, nlosProb] = gnssContextModel(c)
    switch c
        case "OpenSky"
            sv=17; hdop=0.9; cno=43; hacc=0.8; pAvail=0.99; sigmaG=0.75; biasSigma=0.03; nlosProb=0.005;
        case "UrbanCanyon"
            sv=8; hdop=3.0; cno=29; hacc=5.0; pAvail=0.88; sigmaG=3.8; biasSigma=0.55; nlosProb=0.08;
        case "Tunnel"
            sv=0; hdop=20; cno=0; hacc=30; pAvail=0.00; sigmaG=20; biasSigma=0; nlosProb=0;
        case "FeaturePoor"
            sv=14; hdop=1.2; cno=39; hacc=1.2; pAvail=0.97; sigmaG=1.0; biasSigma=0.06; nlosProb=0.01;
        otherwise
            sv=10; hdop=2; cno=35; hacc=2; pAvail=0.9; sigmaG=2; biasSigma=0.1; nlosProb=0.02;
    end
end

function [pAvail, sigmaL, sigmaYaw, scoreMean, featMean, outlierProb] = lidarContextModel(c)
    switch c
        case "OpenSky"
            pAvail=0.995; sigmaL=0.25; sigmaYaw=deg2rad(0.8); scoreMean=0.91; featMean=430; outlierProb=0.005;
        case "UrbanCanyon"
            pAvail=0.995; sigmaL=0.45; sigmaYaw=deg2rad(1.3); scoreMean=0.78; featMean=330; outlierProb=0.025;
        case "Tunnel"
            pAvail=0.995; sigmaL=0.30; sigmaYaw=deg2rad(1.0); scoreMean=0.86; featMean=390; outlierProb=0.008;
        case "FeaturePoor"
            pAvail=0.98; sigmaL=1.9; sigmaYaw=deg2rad(4.0); scoreMean=0.33; featMean=80; outlierProb=0.06;
        otherwise
            pAvail=0.99; sigmaL=0.7; sigmaYaw=deg2rad(2); scoreMean=0.6; featMean=200; outlierProb=0.02;
    end
end

%% ========================================================================
function T = normalizeDatasetTypes(T)
% writetable stores logical columns as 0/1, so a CSV round-trip turns the
% availability flags into doubles. Restore the types the filters expect.
    T.context = string(T.context);
    T.gnss_available = logical(T.gnss_available);
    T.lidar_available = logical(T.lidar_available);
end

%% ========================================================================
%  UNIFIED PARTICLE FILTER (all ablation variants share this code path)
%% ========================================================================
function out = runPF(T, cfg, opt)
% opt.useReliability      schema features -> ontology reliability -> R
% opt.useGuidedProposal   measurement-informed Gaussian proposal
% opt.useAdaptiveN        context/ESS-adaptive particle budget
% opt.N                   particle count when ~useAdaptiveN
% opt.resampleESS         ESS/N resampling threshold
%
% With all three flags false this reduces exactly to a fixed-N bootstrap
% particle filter with fixed measurement covariance, so the ablation rows
% differ only by the components under test.

    K = height(T);
    ruleNames = ontologyRuleNames();
    nRules = numel(ruleNames);

    if opt.useAdaptiveN
        N = cfg.Nmax;
    else
        N = opt.N;
    end

    [mu0, v0] = initialPoseFromData(T);
    particles = initializeParticles(mu0, v0, N, cfg);
    weights = ones(N,1)/N;

    est = nan(K,4);
    ess = nan(K,1);
    nParticles = nan(K,1);
    stepTime = nan(K,1);
    resampled = false(K,1);
    resizedN = false(K,1);
    relGnss = nan(K,1); relLidar = nan(K,1); relLoc = nan(K,1);
    sigGnss = nan(K,1); sigLidarPos = nan(K,1); sigLidarYaw = nan(K,1);
    ruleFired = false(K,nRules);
    measDim = zeros(K,1);
    prevEssRatio = 1.0;

    Q = diag([cfg.qPos^2, cfg.qPos^2, cfg.qYaw^2, cfg.qVel^2]);
    totalTic = tic;

    for k = 1:K
        st = tic;

        % ---- (1) ontology reasoning ------------------------------------
        if opt.useReliability
            [rG, rL, rLoc, rf] = ontologyReasoner(T, k);
            relGnss(k) = rG; relLidar(k) = rL; relLoc(k) = rLoc;
            ruleFired(k,1:numel(rf)) = rf;
        else
            rG = NaN; rL = NaN; rLoc = NaN;
        end

        % ---- (2) adaptive particle budget ------------------------------
        if opt.useAdaptiveN
            Ntarget = adaptiveParticleBudget(rG, rL, rLoc, prevEssRatio, cfg);
            if abs(Ntarget-N)/max(N,1) > cfg.particleChangeThreshold
                idx = systematicResample(weights, Ntarget);
                particles = particles(idx,:);
                % Roughening only when the population is expanded, to avoid
                % duplicating identical particles after up-sampling.
                if Ntarget > N
                    particles(:,1:2) = particles(:,1:2) + cfg.rougheningPos*randn(Ntarget,2);
                    particles(:,3) = wrapPiLocal(particles(:,3) + cfg.rougheningYaw*randn(Ntarget,1));
                end
                N = Ntarget;
                weights = ones(N,1)/N;
                resizedN(k) = true;
            end
        end
        nParticles(k) = N;

        % ---- (3) measurement model -------------------------------------
        if opt.useReliability
            [z,H,R,angleRows,info] = buildOntologyMeasurement(T, k, rG, rL, cfg);
        else
            [z,H,R,angleRows,info] = buildFixedMeasurement(T, k, cfg);
        end
        sigGnss(k) = info.sigmaG;
        sigLidarPos(k) = info.sigmaLP;
        sigLidarYaw(k) = info.sigmaLY;
        ruleFired(k,9)  = info.gnssFloorActive;
        ruleFired(k,10) = info.lidarFloorActive;
        measDim(k) = numel(z);

        % ---- (4) propagate + weight ------------------------------------
        if k == 1
            % Initial population already drawn: likelihood update only.
            if ~isempty(z)
                logw = log(weights+realmin) + measurementLogLikelihood(particles,z,H,R,angleRows);
                weights = normalizeLogWeights(logw);
            end
        elseif opt.useGuidedProposal && ~isempty(z)
            % Approximately optimal proposal q(x_k | x_{k-1}, u_k, z_k, O)
            % for a Gaussian transition and reliability-adapted absolute
            % pose measurements, linearized around each predicted particle.
            priorMean = transitionMean(particles, T.speed_meas(k), T.yaw_rate_meas(k), cfg.dt);

            S = H*Q*H' + R;
            S = (S+S')/2 + 1e-10*eye(size(S));
            Kq = (Q*H')/S;
            Pq = Q - Kq*H*Q;
            Pq = (Pq+Pq')/2 + 1e-10*eye(4);

            innov = repmat(z(:)',N,1) - priorMean*H';
            for ar = angleRows(:)'
                innov(:,ar) = wrapPiLocal(innov(:,ar));
            end
            proposalMean = priorMean + innov*Kq';
            Lq = chol(Pq, 'lower');
            particles = proposalMean + randn(N,4)*Lq';
            particles(:,3) = wrapPiLocal(particles(:,3));
            particles(:,4) = max(0, particles(:,4));

            % Importance correction for the measurement-informed proposal:
            % w_k  proportional to  w_{k-1} * p(z_k | x_{k-1}, u_k)
            Ls = chol(S, 'lower');
            whitened = innov / Ls';
            logPredLike = -0.5*(sum(whitened.^2,2) + 2*sum(log(diag(Ls))) + numel(z)*log(2*pi));
            weights = normalizeLogWeights(log(weights+realmin) + logPredLike);
        else
            % Bootstrap step: sample the transition, weight by likelihood.
            priorMean = transitionMean(particles, T.speed_meas(k), T.yaw_rate_meas(k), cfg.dt);
            Lq = chol(Q + 1e-12*eye(4), 'lower');
            particles = priorMean + randn(N,4)*Lq';
            particles(:,3) = wrapPiLocal(particles(:,3));
            particles(:,4) = max(0, particles(:,4));

            logw = log(weights+realmin);
            if ~isempty(z)
                logw = logw + measurementLogLikelihood(particles,z,H,R,angleRows);
            end
            weights = normalizeLogWeights(logw);
        end

        est(k,:) = weightedStateMean(particles, weights);
        ess(k) = 1/sum(weights.^2);
        prevEssRatio = ess(k)/N;

        % ---- (5) resampling (identical rule for every variant) ---------
        if ess(k) < opt.resampleESS*N
            idx = systematicResample(weights, N);
            particles = particles(idx,:);
            weights = ones(N,1)/N;
            resampled(k) = true;
            prevEssRatio = 1.0;
        end
        stepTime(k) = toc(st);
    end

    out.est = est;
    out.ess = ess;
    out.nParticles = nParticles;
    out.stepTime_s = stepTime;
    out.resampled = resampled;
    out.resizedN = resizedN;
    out.totalRuntime_s = toc(totalTic);
    out.relGnss = relGnss;
    out.relLidar = relLidar;
    out.relLocalization = relLoc;
    out.sigmaGnss = sigGnss;
    out.sigmaLidarPos = sigLidarPos;
    out.sigmaLidarYaw = sigLidarYaw;
    out.ruleFired = ruleFired;
    out.ruleNames = ruleNames;
    out.measDim = measDim;
    out.opt = opt;
end

%% ========================================================================
%  ONTOLOGY / SCHEMA LAYER
%% ========================================================================
function names = ontologyRuleNames()
% Rule identifiers used by the "what did the ontology actually do" plot and
% by results/ontology_trace.csv. R1-R8 are asserted by the reasoner, R9-R10
% by the covariance-floor axiom in the measurement model.
    names = [ ...
        "R1  GnssObservation absent      -> rG := 0"
        "R2  UrbanCanyon multipath       -> rG := 0.55 rG"
        "R3  Tunnel GNSS denied          -> rG := 0"
        "R4  OpenSky nominal             -> rG := min(1, 1.08 rG)"
        "R5  LidarObservation absent     -> rL := 0"
        "R6  FeaturePoor degradation     -> rL := 0.45 rL"
        "R7  Tunnel structure-rich       -> rL := min(1, 1.05 rL)"
        "R8  DeadReckoningOnly state     (rLoc < 0.05)"
        "R9  GNSS covariance floor active  (sigma_G  = floor)"
        "R10 LiDAR covariance floor active (sigma_LP = floor)"];
end

function [rG, rL, rLoc, rf] = ontologyReasoner(T, k)
% Lightweight rule engine corresponding to the research ontology concept.
% In a full OWL/SWRL implementation these scores would be produced by the
% ontology / knowledge graph. Here the rules are explicit MATLAB code so
% the experiment stays reproducible without an external reasoner.
%
% Schema features -> evidence scores -> context rules -> reliability.
% rf(i) is true when rule Ri fired at step k.

    rf = false(1,8);
    c = string(T.context(k));

    % ---- GNSS schema -> reliability evidence -------------------------
    if ~T.gnss_available(k)
        rG = 0;
        rf(1) = true;
    else
        svScore   = clamp01((T.numSV(k)-4)/12);
        hdopScore = exp(-0.38*max(T.hdop(k)-0.8,0));
        cnoScore  = clamp01((T.cn0_mean(k)-20)/25);
        haccScore = exp(-T.gnss_hAcc(k)/6.0);
        rG = 0.25*svScore + 0.30*hdopScore + 0.30*cnoScore + 0.15*haccScore;

        if c == "UrbanCanyon"
            rG = 0.55*rG;              % likely NLOS/multipath context
            rf(2) = true;
        elseif c == "Tunnel"
            rG = 0;
            rf(3) = true;
        elseif c == "OpenSky"
            rG = min(1,1.08*rG);
            rf(4) = true;
        end
    end

    % ---- LiDAR scan-matching schema -> reliability evidence ----------
    if ~T.lidar_available(k)
        rL = 0;
        rf(5) = true;
    else
        scoreEvidence = clamp01(T.lidar_score(k));
        featEvidence  = clamp01((T.lidar_feature_count(k)-50)/400);
        rL = 0.67*scoreEvidence + 0.33*featEvidence;
        if c == "FeaturePoor"
            rL = 0.45*rL;
            rf(6) = true;
        elseif c == "Tunnel"
            rL = min(1,1.05*rL);
            rf(7) = true;
        end
    end

    % ---- Reliability of at least one independent localization source --
    rLoc = 1 - (1-rG)*(1-rL);
    rG = clamp01(rG); rL = clamp01(rL); rLoc = clamp01(rLoc);
    if rLoc < 0.05
        rf(8) = true;
    end
end

function [svScore, hdopScore, cnoScore, haccScore] = gnssEvidenceScores(numSV, hdop, cn0, hAcc)
% Exposed separately so the visualization can draw exactly the same
% feature -> evidence mapping that the reasoner uses.
    svScore   = clamp01((numSV-4)/12);
    hdopScore = exp(-0.38*max(hdop-0.8,0));
    cnoScore  = clamp01((cn0-20)/25);
    haccScore = exp(-hAcc/6.0);
end

function Ntarget = adaptiveParticleBudget(rG, rL, rLoc, prevEssRatio, cfg)
% N_k = f(context, sensor reliability, ESS). Difficult context or a
% degenerate particle set buys more particles; a confidently observed
% state releases them.
    bestRel = max(rG,rL);
    difficulty = 1 - bestRel;
    degeneracy = clamp01((0.75-prevEssRatio)/0.75);
    lowOverallReliability = 1 - rLoc;
    demand = clamp01(0.60*difficulty + 0.25*degeneracy + 0.15*lowOverallReliability);
    Ntarget = round(cfg.Nmin + (cfg.Nmax-cfg.Nmin)*(demand^0.85));
    Ntarget = max(cfg.Nmin,min(cfg.Nmax,Ntarget));
    % Round to a convenient multiple to reduce frequent resizing.
    Ntarget = 50*round(Ntarget/50);
end

function [z,H,R,angleRows,info] = buildOntologyMeasurement(T,k,rG,rL,cfg)
% Reliability-adapted measurement model. Low reliability widens the
% covariance instead of hard-rejecting the measurement, so no hypothesis is
% deleted; only its influence is attenuated.
    z = []; H = []; R = []; angleRows = [];
    info = struct('sigmaG',NaN,'sigmaLP',NaN,'sigmaLY',NaN, ...
                  'gnssFloorActive',false,'lidarFloorActive',false);

    if T.gnss_available(k) && rG > cfg.reliabilityGate
        raw = cfg.gnssSigmaBase + cfg.gnssSigmaGain*(1-rG)^2;
        sigmaG = max(cfg.gnssSigmaFloor, raw);
        info.sigmaG = sigmaG;
        info.gnssFloorActive = raw < cfg.gnssSigmaFloor;
        z = [z; T.gnss_x(k); T.gnss_y(k)];
        H = [H; 1 0 0 0; 0 1 0 0];
        R = blkdiag(R, sigmaG^2*eye(2));
    end

    if T.lidar_available(k) && rL > cfg.reliabilityGate
        rawP = cfg.lidarPosSigmaBase + cfg.lidarPosSigmaGain*(1-rL)^2;
        sigmaLP = max(cfg.lidarPosSigmaFloor, rawP);
        sigmaLY = max(cfg.lidarYawSigmaFloor, cfg.lidarYawSigmaBase + cfg.lidarYawSigmaGain*(1-rL)^2);
        info.sigmaLP = sigmaLP;
        info.sigmaLY = sigmaLY;
        info.lidarFloorActive = rawP < cfg.lidarPosSigmaFloor;
        oldm = numel(z);
        z = [z; T.lidar_x(k); T.lidar_y(k); T.lidar_yaw(k)];
        H = [H; 1 0 0 0; 0 1 0 0; 0 0 1 0];
        R = blkdiag(R, diag([sigmaLP^2 sigmaLP^2 sigmaLY^2]));
        angleRows = [angleRows, oldm+3];
    end
end

function [z,H,R,angleRows,info] = buildFixedMeasurement(T,k,cfg)
% Baseline: fixed, hand-tuned measurement covariance, no ontology.
    z = []; H = []; R = []; angleRows = [];
    info = struct('sigmaG',NaN,'sigmaLP',NaN,'sigmaLY',NaN, ...
                  'gnssFloorActive',false,'lidarFloorActive',false);

    if T.gnss_available(k)
        info.sigmaG = cfg.baseGnssSigma;
        z = [z; T.gnss_x(k); T.gnss_y(k)];
        H = [H; 1 0 0 0; 0 1 0 0];
        R = blkdiag(R, cfg.baseGnssSigma^2*eye(2));
    end

    if T.lidar_available(k)
        info.sigmaLP = cfg.baseLidarPosSigma;
        info.sigmaLY = cfg.baseLidarYawSigma;
        oldm = numel(z);
        z = [z; T.lidar_x(k); T.lidar_y(k); T.lidar_yaw(k)];
        H = [H; 1 0 0 0; 0 1 0 0; 0 0 1 0];
        R = blkdiag(R, diag([cfg.baseLidarPosSigma^2 cfg.baseLidarPosSigma^2 cfg.baseLidarYawSigma^2]));
        angleRows = [angleRows, oldm+3];
    end
end

%% ========================================================================
%  PF PRIMITIVES
%% ========================================================================
function priorMean = transitionMean(particles, speedMeas, yawRateMeas, dt)
    priorMean = particles;
    % Blend velocity state with wheel/odometry speed measurement.
    priorMean(:,4) = 0.88*particles(:,4) + 0.12*speedMeas;
    priorMean(:,3) = wrapPiLocal(particles(:,3) + yawRateMeas*dt);
    priorMean(:,1) = particles(:,1) + priorMean(:,4).*cos(priorMean(:,3))*dt;
    priorMean(:,2) = particles(:,2) + priorMean(:,4).*sin(priorMean(:,3))*dt;
end

function particles = initializeParticles(mu0,v0,N,cfg)
    particles = zeros(N,4);
    particles(:,1) = mu0(1) + cfg.initPosStd*randn(N,1);
    particles(:,2) = mu0(2) + cfg.initPosStd*randn(N,1);
    particles(:,3) = wrapPiLocal(mu0(3) + cfg.initYawStd*randn(N,1));
    particles(:,4) = max(0, v0 + cfg.initVelStd*randn(N,1));
end

function [mu0,v0] = initialPoseFromData(T)
    kL = find(T.lidar_available,1,'first');
    kG = find(T.gnss_available,1,'first');
    if ~isempty(kL)
        mu0 = [T.lidar_x(kL), T.lidar_y(kL), T.lidar_yaw(kL)];
    elseif ~isempty(kG)
        mu0 = [T.gnss_x(kG), T.gnss_y(kG), 0];
    else
        mu0 = [T.true_x(1), T.true_y(1), T.true_yaw(1)];
    end
    v0 = T.speed_meas(1);
end

function logL = measurementLogLikelihood(particles,z,H,R,angleRows)
    innov = repmat(z(:)',size(particles,1),1) - particles*H';
    for ar = angleRows(:)'
        innov(:,ar) = wrapPiLocal(innov(:,ar));
    end
    L = chol((R+R')/2 + 1e-12*eye(size(R)), 'lower');
    W = innov/L';
    q = sum(W.^2,2);
    logdetR = 2*sum(log(diag(L)));
    m = numel(z);
    logL = -0.5*(q + logdetR + m*log(2*pi));
end

function w = normalizeLogWeights(logw)
    logw = logw - max(logw);
    w = exp(logw);
    s = sum(w);
    if ~isfinite(s) || s <= realmin
        w = ones(size(w))/numel(w);
    else
        w = w/s;
    end
end

function mu = weightedStateMean(p,w)
    mu = zeros(1,4);
    mu(1) = sum(w.*p(:,1));
    mu(2) = sum(w.*p(:,2));
    mu(3) = atan2(sum(w.*sin(p(:,3))), sum(w.*cos(p(:,3))));
    mu(4) = sum(w.*p(:,4));
end

function idx = systematicResample(w, Nout)
    w = w(:)/sum(w);
    cdf = cumsum(w);
    cdf(end)=1;
    u0 = rand/Nout;
    u = u0 + (0:Nout-1)'/Nout;
    idx = zeros(Nout,1);
    j=1;
    for i=1:Nout
        while u(i) > cdf(j)
            j=j+1;
        end
        idx(i)=j;
    end
end

function y = wrapPiLocal(x)
    y = mod(x+pi,2*pi)-pi;
end

function y = clamp01(x)
    y = max(0,min(1,x));
end

function q = percentileLocal(x,p)
    x=x(isfinite(x));
    if isempty(x), q=NaN; return; end
    x=sort(x(:));
    r=1+(numel(x)-1)*p/100;
    lo=floor(r); hi=ceil(r);
    if lo==hi
        q=x(lo);
    else
        q=x(lo)+(r-lo)*(x(hi)-x(lo));
    end
end

%% ========================================================================
%  METRICS
%% ========================================================================
function e = positionError(T, out)
    e = hypot(out.est(:,1)-T.true_x, out.est(:,2)-T.true_y);
end

function m = runMetrics(T, out)
    e = positionError(T, out);
    m.rmse = sqrt(mean(e.^2,'omitnan'));
    m.p95 = percentileLocal(e,95);
    m.failure_pct = 100*mean(e>5,'omitnan');
    m.meanN = mean(out.nParticles,'omitnan');
    m.essRatio = mean(out.ess./out.nParticles,'omitnan');
    m.essRatioP10 = percentileLocal(out.ess./out.nParticles,10);
    m.runtime_s = out.totalRuntime_s;
    m.stepTime_ms = 1000*mean(out.stepTime_s,'omitnan');
    m.resampleRate_pct = 100*mean(out.resampled);
end

function [summaryTbl, contextTbl] = computeMetrics(T, runs)
    n = numel(runs);
    Algorithm = strings(n,1);
    PositionRMSE_m = zeros(n,1); P95Error_m = zeros(n,1);
    FailureRate_pct = zeros(n,1); MeanParticles = zeros(n,1);
    MeanESSRatio = zeros(n,1); P10ESSRatio = zeros(n,1);
    Runtime_s = zeros(n,1); MeanStepTime_ms = zeros(n,1);
    ResampleRate_pct = zeros(n,1);

    for i = 1:n
        m = runMetrics(T, runs(i).out);
        Algorithm(i) = runs(i).name;
        PositionRMSE_m(i) = m.rmse;
        P95Error_m(i) = m.p95;
        FailureRate_pct(i) = m.failure_pct;
        MeanParticles(i) = m.meanN;
        MeanESSRatio(i) = m.essRatio;
        P10ESSRatio(i) = m.essRatioP10;
        Runtime_s(i) = m.runtime_s;
        MeanStepTime_ms(i) = m.stepTime_ms;
        ResampleRate_pct(i) = m.resampleRate_pct;
    end
    summaryTbl = table(Algorithm,PositionRMSE_m,P95Error_m,FailureRate_pct, ...
        MeanParticles,MeanESSRatio,P10ESSRatio,Runtime_s,MeanStepTime_ms,ResampleRate_pct);

    % ---- context-wise ------------------------------------------------
    ctx = string(T.context);
    contexts = unique(ctx,'stable');
    nC = numel(contexts);
    Context = contexts;
    contextTbl = table(Context);
    for i = 1:n
        e = positionError(T, runs(i).out);
        rmseCol = zeros(nC,1); p95Col = zeros(nC,1); nCol = zeros(nC,1);
        for j = 1:nC
            mask = ctx==contexts(j);
            rmseCol(j) = sqrt(mean(e(mask).^2,'omitnan'));
            p95Col(j) = percentileLocal(e(mask),95);
            nCol(j) = mean(runs(i).out.nParticles(mask),'omitnan');
        end
        nm = char(matlab.lang.makeValidName(runs(i).name));
        contextTbl.([nm '_RMSE_m']) = rmseCol;
        contextTbl.([nm '_P95_m']) = p95Col;
        contextTbl.([nm '_MeanN']) = nCol;
    end
end

function ablationTbl = computeAblationTable(T, runs)
% One row per ablation stage with the delta relative to PF-Base.
    n = numel(runs);
    Variant = strings(n,1); Description = strings(n,1);
    Reliability = strings(n,1); GuidedProposal = strings(n,1); AdaptiveN = strings(n,1);
    RMSE_m = zeros(n,1); P95_m = zeros(n,1); MeanN = zeros(n,1);
    MeanESSRatio = zeros(n,1); StepTime_ms = zeros(n,1);

    yn = @(b) string(char(79 + 9*~b));   % 'O' when true, 'X' when false
    for i = 1:n
        m = runMetrics(T, runs(i).out);
        o = runs(i).out.opt;
        Variant(i) = runs(i).name;
        Description(i) = string(runs(i).label);
        Reliability(i) = yn(o.useReliability);
        GuidedProposal(i) = yn(o.useGuidedProposal);
        AdaptiveN(i) = yn(o.useAdaptiveN);
        RMSE_m(i) = m.rmse; P95_m(i) = m.p95; MeanN(i) = m.meanN;
        MeanESSRatio(i) = m.essRatio; StepTime_ms(i) = m.stepTime_ms;
    end
    RMSE_gain_pct = 100*(RMSE_m(1)-RMSE_m)/RMSE_m(1);
    P95_gain_pct = 100*(P95_m(1)-P95_m)/P95_m(1);
    Particle_saving_pct = 100*(MeanN(1)-MeanN)/MeanN(1);
    % Marginal contribution of the component added in this row.
    RMSE_step_gain_pct = [0; 100*(RMSE_m(1:end-1)-RMSE_m(2:end))./RMSE_m(1)];

    ablationTbl = table(Variant,Reliability,GuidedProposal,AdaptiveN, ...
        RMSE_m,RMSE_gain_pct,RMSE_step_gain_pct,P95_m,P95_gain_pct, ...
        MeanN,Particle_saving_pct,MeanESSRatio,StepTime_ms,Description);
end

function tr = ontologyTraceTable(T, out)
% Per-step record of what the schema/ontology layer produced and what the
% filter then used. This is the machine-readable version of
% ontology_applied.png.
    tr = table(T.time_s, string(T.context), ...
        T.numSV, T.hdop, T.cn0_mean, T.gnss_hAcc, ...
        T.lidar_score, T.lidar_feature_count, ...
        out.relGnss, out.relLidar, out.relLocalization, ...
        out.sigmaGnss, out.sigmaLidarPos, rad2deg(out.sigmaLidarYaw), ...
        out.nParticles, out.ess, out.ess./out.nParticles, out.measDim, ...
        'VariableNames', {'time_s','context','numSV','hdop','cn0_mean','gnss_hAcc', ...
        'lidar_score','lidar_feature_count','rel_gnss','rel_lidar','rel_localization', ...
        'sigma_gnss_m','sigma_lidar_pos_m','sigma_lidar_yaw_deg', ...
        'n_particles','ess','ess_ratio','meas_dim'});
    rn = out.ruleNames;
    for i = 1:numel(rn)
        key = extractBefore(rn(i)," ");
        tr.(char("rule_"+key)) = double(out.ruleFired(:,i));
    end
end

%% ========================================================================
%  MONTE CARLO
%% ========================================================================
function mcOut = monteCarloStudy(cfg, mode, T0)
% mode = "filterSeed"            : fixed dataset, filter seed varies.
%                                  Isolates PF sampling randomness.
% mode = "datasetAndFilterSeed"  : dataset and filter seed both vary.
%                                  Also covers scenario/measurement noise.
%
% Runtime is aggregated as a MEDIAN: a single MATLAB tic/toc is dominated
% by JIT warm-up and CPU scheduling, so single-run timings are not
% comparable between algorithms.

    V = pfVariants(cfg);
    nV = numel(V);
    R = cfg.mcRuns;
    ctx0 = unique(string(T0.context),'stable');
    nC = numel(ctx0);

    rmse = nan(R,nV); p95 = nan(R,nV); fail = nan(R,nV);
    meanN = nan(R,nV); essR = nan(R,nV); essP10 = nan(R,nV);
    runtime = nan(R,nV); stepMs = nan(R,nV); resampPct = nan(R,nV);
    ctxRmse = nan(R,nC,nV);

    for r = 1:R
        if mode == "filterSeed"
            Tr = T0;
        else
            rng(cfg.datasetSeed + 1000*r, 'twister');
            Tr = generateUrbanNavLikeDataset(cfg);
        end
        ctxR = string(Tr.context);
        fs = cfg.filterSeed + r - 1;
        for i = 1:nV
            rng(fs, 'twister');
            o = runPF(Tr, cfg, V(i).opt);
            m = runMetrics(Tr, o);
            rmse(r,i)=m.rmse; p95(r,i)=m.p95; fail(r,i)=m.failure_pct;
            meanN(r,i)=m.meanN; essR(r,i)=m.essRatio; essP10(r,i)=m.essRatioP10;
            runtime(r,i)=m.runtime_s; stepMs(r,i)=m.stepTime_ms;
            resampPct(r,i)=m.resampleRate_pct;
            e = positionError(Tr,o);
            for j = 1:nC
                mask = ctxR==ctx0(j);
                if any(mask)
                    ctxRmse(r,j,i) = sqrt(mean(e(mask).^2,'omitnan'));
                end
            end
        end
        if mod(r,5)==0 || r==R
            msg = '';
            for i = 1:nV
                msg = [msg sprintf('%s=%.3f  ', V(i).name, rmse(r,i))]; %#ok<AGROW>
            end
            fprintf('         run %2d/%2d  RMSE %s\n', r, R, msg);
        end
    end

    Variant = [V.name]';
    mcOut.summary = table(Variant, ...
        mean(rmse,1)', std(rmse,0,1)', ...
        mean(p95,1)',  std(p95,0,1)', ...
        mean(fail,1)', mean(meanN,1)', std(meanN,0,1)', ...
        mean(essR,1)', std(essR,0,1)', mean(essP10,1)', ...
        median(runtime,1)', mean(runtime,1)', std(runtime,0,1)', ...
        median(stepMs,1)', mean(resampPct,1)', ...
        'VariableNames', {'Variant','RMSE_mean_m','RMSE_std_m','P95_mean_m','P95_std_m', ...
        'Failure_mean_pct','MeanN_mean','MeanN_std','ESSRatio_mean','ESSRatio_std', ...
        'ESSRatio_p10_mean','Runtime_median_s','Runtime_mean_s','Runtime_std_s', ...
        'StepTime_median_ms','ResampleRate_mean_pct'});

    ctxTbl = table(ctx0,'VariableNames',{'Context'});
    for i = 1:nV
        nm = char(matlab.lang.makeValidName(V(i).name));
        ctxTbl.([nm '_RMSE_mean_m']) = squeeze(mean(ctxRmse(:,:,i),1))';
        ctxTbl.([nm '_RMSE_std_m'])  = squeeze(std(ctxRmse(:,:,i),0,1))';
    end
    mcOut.context = ctxTbl;

    mcOut.mode = mode;
    mcOut.runs = R;
    mcOut.variantNames = Variant;
    mcOut.rmse = rmse; mcOut.p95 = p95; mcOut.meanN = meanN;
    mcOut.essRatio = essR; mcOut.runtime = runtime; mcOut.stepMs = stepMs;
    mcOut.ctxRmse = ctxRmse; mcOut.contexts = ctx0;
end

function printPairedImprovement(mcOut)
% Paired per-run comparison of the full method against PF-Base. Paired
% differences remove the run-to-run scenario difficulty, so this is the
% right statistic when the same seed drives both filters. No Statistics
% Toolbox is used.
    dR = mcOut.rmse(:,1) - mcOut.rmse(:,end);
    dP = mcOut.p95(:,1)  - mcOut.p95(:,end);
    relR = 100*dR./mcOut.rmse(:,1);
    relP = 100*dP./mcOut.p95(:,1);
    R = size(mcOut.rmse,1);
    se = std(dR,0,1)/sqrt(R);
    fprintf('  Paired PF-Base -> %s over %d runs:\n', mcOut.variantNames(end), R);
    fprintf('    dRMSE = %+.4f +/- %.4f m  (%+.1f%% +/- %.1f%%), wins %d/%d, |d|/SE = %.1f\n', ...
        mean(dR), std(dR,0,1), mean(relR), std(relR,0,1), sum(dR>0), R, abs(mean(dR))/max(se,eps));
    fprintf('    dP95  = %+.4f +/- %.4f m  (%+.1f%% +/- %.1f%%), wins %d/%d\n', ...
        mean(dP), std(dP,0,1), mean(relP), std(relP,0,1), sum(dP>0), R);
    fprintf('    mean N: %.0f -> %.0f (%.1f%% fewer particles)\n', ...
        mean(mcOut.meanN(:,1)), mean(mcOut.meanN(:,end)), ...
        100*(mean(mcOut.meanN(:,1))-mean(mcOut.meanN(:,end)))/mean(mcOut.meanN(:,1)));
    fprintf('    median runtime: %.3f s -> %.3f s (%+.1f%%)\n', ...
        median(mcOut.runtime(:,1)), median(mcOut.runtime(:,end)), ...
        100*(median(mcOut.runtime(:,end))-median(mcOut.runtime(:,1)))/median(mcOut.runtime(:,1)));
    disp(' ');
end

%% ========================================================================
%  PLOT HELPERS
%% ========================================================================
function [cats, cmap] = contextColors(T)
% Fixed colour per environment class so every figure is read the same way.
    cats = unique(string(T.context),'stable');
    cmap = zeros(numel(cats),3);
    for i = 1:numel(cats)
        switch cats(i)
            case "OpenSky",     cmap(i,:) = [0.16 0.55 0.85];
            case "UrbanCanyon", cmap(i,:) = [0.92 0.58 0.10];
            case "Tunnel",      cmap(i,:) = [0.32 0.32 0.38];
            case "FeaturePoor", cmap(i,:) = [0.76 0.24 0.52];
            otherwise,          cmap(i,:) = [0.30 0.66 0.36];
        end
    end
end

function segs = contextSegments(T)
% Contiguous runs of a single context: segs(i) = struct(name, i0, i1).
    c = string(T.context);
    edges = [1; find(c(2:end)~=c(1:end-1))+1; height(T)+1];
    segs = struct('name',{},'i0',{},'i1',{});
    for i = 1:numel(edges)-1
        segs(i).name = c(edges(i));
        segs(i).i0 = edges(i);
        segs(i).i1 = edges(i+1)-1;
    end
end

function addContextBands(ax, T, doLabel)
% Translucent environment bands behind a time-series plot.
    if nargin < 3, doLabel = true; end
    [cats, cmap] = contextColors(T);
    segs = contextSegments(T);
    yl = ylim(ax);
    hold(ax,'on');
    for i = 1:numel(segs)
        ci = find(cats==segs(i).name,1);
        x1 = T.time_s(segs(i).i0);
        x2 = T.time_s(segs(i).i1);
        p = patch(ax,[x1 x2 x2 x1],[yl(1) yl(1) yl(2) yl(2)], cmap(ci,:), ...
            'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
        uistack(p,'bottom');
        xline(ax,x2,':','Color',[0.5 0.5 0.5],'HandleVisibility','off');
        if doLabel
            text(ax,0.5*(x1+x2), yl(2)-0.02*(yl(2)-yl(1)), char(segs(i).name), ...
                'HorizontalAlignment','center','VerticalAlignment','top', ...
                'FontSize',7,'Color',0.55*cmap(ci,:),'FontWeight','bold', ...
                'HandleVisibility','off');
        end
    end
    ylim(ax,yl);
end

function drawBox(ax,xc,yc,w,h,txt,faceColor)
    rectangle(ax,'Position',[xc-w/2, yc-h/2, w, h],'Curvature',0.18, ...
        'FaceColor',faceColor,'EdgeColor',0.55*faceColor,'LineWidth',0.8);
    text(ax,xc,yc,txt,'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',7,'Interpreter','none');
end

function drawArrow(ax,x1,y1,x2,y2)
    plot(ax,[x1 x2],[y1 y2],'-','Color',[0.40 0.40 0.45],'LineWidth',0.9, ...
        'HandleVisibility','off');
    d = [x2-x1, y2-y1];
    L = hypot(d(1),d(2));
    if L < eps, return; end
    u = d/L; n = [-u(2), u(1)];
    hl = 1.6; hw = 0.9;
    tip = [x2 y2]; b = tip - hl*u;
    patch(ax,'XData',[tip(1), b(1)+hw*n(1), b(1)-hw*n(1)], ...
             'YData',[tip(2), b(2)+hw*n(2), b(2)-hw*n(2)], ...
             'FaceColor',[0.40 0.40 0.45],'EdgeColor','none','HandleVisibility','off');
end

function h = barWithError(ax, labels, vals, errs, faceColor)
    h = bar(ax, categorical(labels, labels), vals, 0.6, 'FaceColor', faceColor);
    if ~isempty(errs) && any(isfinite(errs))
        hold(ax,'on');
        errorbar(ax, h.XEndPoints, vals, errs, 'k', 'linestyle','none','LineWidth',0.9);
    end
    grid(ax,'on');
end

%% ========================================================================
%  FIGURE 1: trajectory / error / budget / reliability
%% ========================================================================
function plotComparison(T, runs, contextTbl, resultDir)
    base = runs(1).out;
    prop = runs(end).out;
    eB = positionError(T,base);
    eP = positionError(T,prop);
    baseName = char(runs(1).name);
    propName = char(runs(end).name);

    f1 = figure('Name','PF comparison','Color','w','Position',[60 60 1400 860]);
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    ax = nexttile;
    plot(ax,T.true_x,T.true_y,'k-','LineWidth',2); hold(ax,'on');
    plot(ax,base.est(:,1),base.est(:,2),'--','LineWidth',1.2,'Color',[0.85 0.33 0.10]);
    plot(ax,prop.est(:,1),prop.est(:,2),'-','LineWidth',1.4,'Color',[0.00 0.45 0.74]);
    axis(ax,'equal'); grid(ax,'on');
    xlabel(ax,'East / local x [m]'); ylabel(ax,'North / local y [m]');
    title(ax,'Trajectory');
    legend(ax,'Ground truth',baseName,propName,'Location','best');

    ax = nexttile;
    plot(ax,T.time_s,eB,'LineWidth',1.0,'Color',[0.85 0.33 0.10]); hold(ax,'on');
    plot(ax,T.time_s,eP,'LineWidth',1.2,'Color',[0.00 0.45 0.74]);
    grid(ax,'on'); xlabel(ax,'Time [s]'); ylabel(ax,'2-D position error [m]');
    title(ax,'Localization error');
    legend(ax,baseName,propName,'Location','northwest');
    addContextBands(ax,T);

    ax = nexttile;
    plot(ax,T.time_s,base.nParticles,'--','LineWidth',1.2,'Color',[0.85 0.33 0.10]);
    hold(ax,'on');
    plot(ax,T.time_s,prop.nParticles,'-','LineWidth',1.4,'Color',[0.00 0.45 0.74]);
    ylim(ax,[0 1.15*max(base.nParticles)]);
    grid(ax,'on'); xlabel(ax,'Time [s]'); ylabel(ax,'Particle count N_k');
    title(ax,sprintf('Sampling budget  (mean %.0f -> %.0f, %.1f%% fewer)', ...
        mean(base.nParticles), mean(prop.nParticles), ...
        100*(mean(base.nParticles)-mean(prop.nParticles))/mean(base.nParticles)));
    legend(ax,'fixed N','adaptive N_k','Location','southwest');
    addContextBands(ax,T);

    ax = nexttile;
    plot(ax,T.time_s,prop.relGnss,'LineWidth',1.1,'Color',[0.47 0.67 0.19]); hold(ax,'on');
    plot(ax,T.time_s,prop.relLidar,'LineWidth',1.1,'Color',[0.49 0.18 0.56]);
    plot(ax,T.time_s,prop.relLocalization,'k-','LineWidth',1.4);
    ylim(ax,[0 1.05]); grid(ax,'on');
    xlabel(ax,'Time [s]'); ylabel(ax,'Reliability score');
    title(ax,'Ontology-inferred reliability');
    legend(ax,'r_G (GNSS)','r_L (LiDAR)','r_{Loc} combined','Location','southwest');
    addContextBands(ax,T);

    title(tl,'Baseline Particle Filter vs. Schema/Ontology-Guided Adaptive Particle Filter');
    exportgraphics(f1,fullfile(resultDir,'trajectory_and_efficiency.png'),'Resolution',180);
end

%% ========================================================================
%  FIGURE 2: diagnostics
%% ========================================================================
function plotDiagnostics(T, runs, contextTbl, resultDir)
    base = runs(1).out;
    prop = runs(end).out;
    eB = positionError(T,base);
    eP = positionError(T,prop);
    baseName = char(runs(1).name);
    propName = char(runs(end).name);

    f2 = figure('Name','PF diagnostics','Color','w','Position',[80 80 1400 860]);
    tl2 = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    ax = nexttile;
    rB = base.ess./base.nParticles;
    rP = prop.ess./prop.nParticles;
    W = 50;   % 5 s moving average; the raw per-step signal is pure hash
    plot(ax,T.time_s,rB,'-','LineWidth',0.4,'Color',[0.85 0.33 0.10 0.10], ...
        'HandleVisibility','off');
    hold(ax,'on');
    plot(ax,T.time_s,rP,'-','LineWidth',0.4,'Color',[0.00 0.45 0.74 0.10], ...
        'HandleVisibility','off');
    plot(ax,T.time_s,movmean(rB,W),'--','LineWidth',1.8,'Color',[0.85 0.33 0.10]);
    plot(ax,T.time_s,movmean(rP,W),'-','LineWidth',1.8,'Color',[0.00 0.45 0.74]);
    ylim(ax,[0 1.05]); grid(ax,'on');
    xlabel(ax,'Time [s]'); ylabel(ax,'ESS / N');
    title(ax,sprintf('Particle-set health, %.0f-s moving average  (mean %.3f vs %.3f at the same threshold)', ...
        W*0.1, mean(rB,'omitnan'), mean(rP,'omitnan')));
    legend(ax,baseName,propName,'Location','southwest');
    addContextBands(ax,T);

    ax = nexttile;
    nm = arrayfun(@(r) matlab.lang.makeValidName(char(r.name)), runs, 'UniformOutput', false);
    M = zeros(height(contextTbl), numel(runs));
    for i = 1:numel(runs)
        M(:,i) = contextTbl.([nm{i} '_RMSE_m']);
    end
    bar(ax, categorical(contextTbl.Context, contextTbl.Context), M);
    grid(ax,'on'); ylabel(ax,'Position RMSE [m]');
    title(ax,'RMSE by environment context');
    legend(ax, cellstr(string({runs.name})), 'Location','northwest');

    ax = nexttile;
    sB = 1000*movmean(base.stepTime_s,25);
    sP = 1000*movmean(prop.stepTime_s,25);
    plot(ax,T.time_s,sB,'--','LineWidth',1.2,'Color',[0.85 0.33 0.10]);
    hold(ax,'on');
    plot(ax,T.time_s,sP,'-','LineWidth',1.3,'Color',[0.00 0.45 0.74]);
    % The first few tenths of a second are JIT warm-up and would otherwise
    % set the y range for the whole plot.
    yTop = 2.2*max([median(sB(50:end)) median(sP(50:end))]);
    ylim(ax,[0 yTop]);
    grid(ax,'on'); xlabel(ax,'Time [s]'); ylabel(ax,'Moving-average step time [ms]');
    title(ax,'Computation time per update (single run: indicative only, see monte\_carlo.png)');
    legend(ax,baseName,propName,'Location','southeast');
    addContextBands(ax,T);

    ax = nexttile;
    edges = linspace(0, max([eB;eP]), 61);
    histogram(ax,eB,edges,'Normalization','probability','FaceColor',[0.85 0.33 0.10],'EdgeColor','none');
    hold(ax,'on');
    histogram(ax,eP,edges,'Normalization','probability','FaceColor',[0.00 0.45 0.74],'EdgeColor','none');
    xline(ax,percentileLocal(eB,95),'--','Color',[0.85 0.33 0.10],'LineWidth',1.4, ...
        'Label',sprintf('P95 %s = %.2f m',baseName,percentileLocal(eB,95)), ...
        'LabelOrientation','horizontal','LabelVerticalAlignment','top', ...
        'FontSize',7,'HandleVisibility','off');
    xline(ax,percentileLocal(eP,95),'-','Color',[0.00 0.45 0.74],'LineWidth',1.4, ...
        'Label',sprintf('P95 %s = %.2f m',propName,percentileLocal(eP,95)), ...
        'LabelOrientation','horizontal','LabelHorizontalAlignment','left', ...
        'LabelVerticalAlignment','middle','FontSize',7,'HandleVisibility','off');
    grid(ax,'on'); xlabel(ax,'2-D position error [m]'); ylabel(ax,'Probability');
    title(ax,'Error distribution (right tail = large-error suppression)');
    legend(ax,baseName,propName,'Location','northeast');

    title(tl2,'Diagnostics');
    exportgraphics(f2,fullfile(resultDir,'diagnostics.png'),'Resolution',180);
end

%% ========================================================================
%  FIGURE 3: 3-D ENVIRONMENT VISUALIZATION
%% ========================================================================
function plotEnvironment3D(T, runs, cfg, resultDir)
% Everything the environment does to the sensors, drawn in 3-D over the
% driven route. x/y is the local ENU ground plane in every panel; the z
% axis carries the environment or algorithm quantity of interest.

    base = runs(1).out;
    prop = runs(end).out;
    eB = positionError(T,base);
    eP = positionError(T,prop);
    [cats, cmap] = contextColors(T);
    nC = numel(cats);
    ctx = string(T.context);

    W = 25;   % 2.5 s smoothing for the fast-varying z quantities
    f = figure('Name','3-D environment','Color','w','Position',[30 30 1700 950]);
    tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    % ---- (1) route over time, coloured by environment class -----------
    ax = nexttile; hold(ax,'on');
    for i = 1:nC
        m = ctx==cats(i);
        xx = T.true_x; yy = T.true_y; zz = T.time_s;
        xx(~m)=NaN; yy(~m)=NaN; zz(~m)=NaN;
        plot3(ax,xx,yy,zz,'-','LineWidth',3.2,'Color',cmap(i,:),'DisplayName',char(cats(i)));
        plot3(ax,T.true_x(m),T.true_y(m),zeros(sum(m),1),'-','LineWidth',2.4, ...
            'Color',[cmap(i,:) 0.35],'HandleVisibility','off');
    end
    gm = T.gnss_available;
    plot3(ax,T.gnss_x(gm),T.gnss_y(gm),T.time_s(gm),'.','MarkerSize',2.5, ...
        'Color',[0.25 0.25 0.25],'DisplayName','GNSS fix');
    setRoute3DAxes(ax,T);
    zlabel(ax,'time [s]');
    title(ax,'(1) route x environment class x time');
    legend(ax,'Location','northwest','FontSize',7);

    % ---- (2) GNSS environment quality --------------------------------
    ax = nexttile;
    scatter3(ax,T.true_x,T.true_y,T.numSV,10,T.cn0_mean,'filled');
    hold(ax,'on');
    plot3(ax,T.true_x,T.true_y,zeros(height(T),1),'-','Color',[0.80 0.80 0.80], ...
        'LineWidth',1.0);
    colormap(ax,parula); cb = colorbar(ax); cb.Label.String = 'C/N_0 [dB-Hz]';
    setRoute3DAxes(ax,T);
    zlabel(ax,'visible satellites');
    title(ax,'(2) GNSS schema: numSV (z), C/N_0 (colour)');

    % ---- (3) LiDAR environment quality -------------------------------
    ax = nexttile;
    scatter3(ax,T.true_x,T.true_y,T.lidar_score,10,T.lidar_feature_count,'filled');
    hold(ax,'on');
    plot3(ax,T.true_x,T.true_y,zeros(height(T),1),'-','Color',[0.80 0.80 0.80], ...
        'LineWidth',1.0);
    colormap(ax,turboLike()); cb = colorbar(ax); cb.Label.String = 'feature count';
    setRoute3DAxes(ax,T); zlim(ax,[0 1]);
    zlabel(ax,'scan-match score');
    title(ax,'(3) LiDAR schema: match score (z), #features (colour)');

    % ---- (4) ontology reliability over the route ---------------------
    % r_G / r_L are only defined when that sensor actually reported, so the
    % raw series is plotted as markers at those epochs and the trend as a
    % smoothed line. Plotting the raw series as a line would draw a wall.
    ax = nexttile; hold(ax,'on');
    rgv = prop.relGnss;  rgv(~T.gnss_available) = NaN;
    rlv = prop.relLidar; rlv(~T.lidar_available) = NaN;
    plot3(ax,T.true_x,T.true_y,rgv,'.','MarkerSize',4,'Color',[0.70 0.84 0.55], ...
        'HandleVisibility','off');
    plot3(ax,T.true_x,T.true_y,rlv,'.','MarkerSize',4,'Color',[0.80 0.66 0.86], ...
        'HandleVisibility','off');
    plot3(ax,T.true_x,T.true_y,movmean(rgv,W,'omitnan'),'-','LineWidth',2.2, ...
        'Color',[0.35 0.60 0.12],'DisplayName','r_G (GNSS epochs)');
    plot3(ax,T.true_x,T.true_y,movmean(rlv,W,'omitnan'),'-','LineWidth',2.2, ...
        'Color',[0.49 0.18 0.56],'DisplayName','r_L (LiDAR epochs)');
    plot3(ax,T.true_x,T.true_y,movmean(prop.relLocalization,W),'k-','LineWidth',1.6, ...
        'DisplayName','r_{Loc} combined');
    setRoute3DAxes(ax,T); zlim(ax,[0 1.05]);
    zlabel(ax,'inferred reliability');
    title(ax,'(4) ontology output along the route');
    legend(ax,'Location','northwest','FontSize',7);

    % ---- (5) adaptive particle budget over the route -----------------
    ax = nexttile; hold(ax,'on');
    for i = 1:nC
        m = ctx==cats(i);
        plot3(ax,T.true_x(m),T.true_y(m),prop.nParticles(m),'.','MarkerSize',8, ...
            'Color',cmap(i,:),'DisplayName',sprintf('%s: mean %.0f', ...
            cats(i), mean(prop.nParticles(m),'omitnan')));
    end
    plot3(ax,T.true_x,T.true_y,base.nParticles,'-','Color',[0.55 0.55 0.55], ...
        'LineWidth',1.4,'DisplayName',sprintf('%s fixed N',char(runs(1).name)));
    setRoute3DAxes(ax,T); zlim(ax,[0 1.08*max(base.nParticles)]);
    zlabel(ax,'particles N_k');
    title(ax,sprintf('(5) where the budget is spent (overall mean %.0f)', ...
        mean(prop.nParticles,'omitnan')));
    legend(ax,'Location','northwest','FontSize',7);

    % ---- (6) error along the route ------------------------------------
    ax = nexttile; hold(ax,'on');
    plot3(ax,T.true_x,T.true_y,eB,'-','LineWidth',0.4,'Color',[0.85 0.33 0.10 0.25], ...
        'HandleVisibility','off');
    plot3(ax,T.true_x,T.true_y,eP,'-','LineWidth',0.4,'Color',[0.00 0.45 0.74 0.25], ...
        'HandleVisibility','off');
    plot3(ax,T.true_x,T.true_y,movmean(eB,W),'-','LineWidth',2.2,'Color',[0.85 0.33 0.10], ...
        'DisplayName',char(runs(1).name));
    plot3(ax,T.true_x,T.true_y,movmean(eP,W),'-','LineWidth',2.2,'Color',[0.00 0.45 0.74], ...
        'DisplayName',char(runs(end).name));
    setRoute3DAxes(ax,T);
    zlabel(ax,'position error [m]');
    title(ax,sprintf('(6) error along the route (%.1f-s moving average)',W*0.1));
    legend(ax,'Location','northwest','FontSize',7);

    title(tl,'3-D environment / sensor context and its effect on the adaptive particle filter');
    exportgraphics(f,fullfile(resultDir,'environment_3d.png'),'Resolution',180);
end

function setRoute3DAxes(ax, T)
% Common framing for every route-based 3-D panel: limits tight to the
% driven area, a ground plane whose x:y proportions are geometrically true
% (the route is ~3.5x longer than it is wide), and a consistent viewpoint.
    padX = 0.03*range(T.true_x);
    padY = 0.10*range(T.true_y);
    xlim(ax,[min(T.true_x)-padX, max(T.true_x)+padX]);
    ylim(ax,[min(T.true_y)-padY, max(T.true_y)+padY]);
    rx = range(xlim(ax));
    ry = range(ylim(ax));
    pbaspect(ax,[1, max(ry/rx,0.30), 0.62]);
    view(ax,-28,26);
    grid(ax,'on'); box(ax,'on');
    xlabel(ax,'x [m]'); ylabel(ax,'y [m]');
    ax.FontSize = 8;
end

function m = turboLike()
% Small perceptual ramp so the figure does not depend on a specific
% MATLAB release providing turbo().
    m = interp1([0 0.25 0.5 0.75 1]', ...
        [0.19 0.07 0.23; 0.13 0.56 0.55; 0.35 0.78 0.35; 0.98 0.75 0.18; 0.72 0.13 0.11], ...
        linspace(0,1,64)');
end

%% ========================================================================
%  FIGURE 4: HOW THE SCHEMA + ONTOLOGY IS APPLIED (mapping / pipeline)
%% ========================================================================
function plotOntologyPipeline(T, prop, cfg, resultDir)
    f = figure('Name','Ontology pipeline','Color','w','Position',[40 40 1560 900]);
    tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    ax = nexttile([1 3]);
    drawOntologyDiagram(ax, cfg);

    % ---- (a) schema feature -> evidence score ------------------------
    ax = nexttile; hold(ax,'on');
    u = linspace(0,1,201)';
    sv   = 20*u;   hd = 10*u;   cn = 50*u;   ha = 30*u;
    [svS, hdS, cnS, haS] = gnssEvidenceScores(sv, hd, cn, ha);
    plot(ax,u,svS,'LineWidth',1.6,'DisplayName','numSV / 20   (w = 0.25)');
    plot(ax,u,hdS,'LineWidth',1.6,'DisplayName','HDOP / 10   (w = 0.30)');
    plot(ax,u,cnS,'LineWidth',1.6,'DisplayName','C/N_0 / 50 dB-Hz  (w = 0.30)');
    plot(ax,u,haS,'LineWidth',1.6,'DisplayName','hAcc / 30 m  (w = 0.15)');
    plot(ax,u,clamp01(0.67*u + 0.33*clamp01((450*u-50)/400)),'k--','LineWidth',1.4, ...
        'DisplayName','LiDAR score+features -> r_L');
    grid(ax,'on'); xlim(ax,[0 1]); ylim(ax,[0 1.02]);
    xlabel(ax,'normalized raw schema feature'); ylabel(ax,'evidence score');
    title(ax,'(a) schema feature -> ontology evidence');
    legend(ax,'Location','southeast','FontSize',7);

    % ---- (b) reliability -> measurement sigma, with the floor --------
    ax = nexttile; hold(ax,'on');
    r = linspace(0,1,201)';
    sgRaw = cfg.gnssSigmaBase + cfg.gnssSigmaGain*(1-r).^2;
    slRaw = cfg.lidarPosSigmaBase + cfg.lidarPosSigmaGain*(1-r).^2;
    sg = max(cfg.gnssSigmaFloor, sgRaw);
    sl = max(cfg.lidarPosSigmaFloor, slRaw);
    mG = sgRaw<=cfg.gnssSigmaFloor;
    mL = slRaw<=cfg.lidarPosSigmaFloor;
    plot(ax,r(mG),sg(mG),'-','LineWidth',6,'Color',[0.78 0.88 0.62], ...
        'DisplayName','R9 GNSS floor region');
    plot(ax,r(mL),sl(mL),'-','LineWidth',6,'Color',[0.82 0.70 0.86], ...
        'DisplayName','R10 LiDAR floor region');
    plot(ax,r,sgRaw,':','LineWidth',1.0,'Color',[0.47 0.67 0.19],'DisplayName','\sigma_G no floor');
    plot(ax,r,slRaw,':','LineWidth',1.0,'Color',[0.49 0.18 0.56],'DisplayName','\sigma_{LP} no floor');
    plot(ax,r,sg,'-','LineWidth',1.8,'Color',[0.47 0.67 0.19],'DisplayName','\sigma_G applied');
    plot(ax,r,sl,'-','LineWidth',1.8,'Color',[0.49 0.18 0.56],'DisplayName','\sigma_{LP} applied');
    yline(ax,cfg.baseGnssSigma,'--','Color',[0.85 0.33 0.10],'LineWidth',1.1, ...
        'Label','baseline \sigma_G = 1.5 m','LabelHorizontalAlignment','left', ...
        'FontSize',7,'HandleVisibility','off');
    yline(ax,cfg.baseLidarPosSigma,'--','Color',[0.60 0.20 0.60],'LineWidth',1.1, ...
        'Label','baseline \sigma_{LP} = 0.7 m','LabelHorizontalAlignment','left', ...
        'FontSize',7,'HandleVisibility','off');
    set(ax,'YScale','log','YTick',[0.2 0.3 0.5 0.7 0.9 1.5 3 5 9]); grid(ax,'on');
    xlim(ax,[0 1]); ylim(ax,[0.15 12]);
    xlabel(ax,'inferred reliability r'); ylabel(ax,'applied \sigma [m]');
    title(ax,'(b) reliability -> measurement covariance (R9/R10 floors)');
    legend(ax,'Location','northeast','FontSize',6.5);

    % ---- (c) 3-D particle-budget rule surface ------------------------
    ax = nexttile;
    dg = linspace(0,1,51);
    dgn = linspace(0,1,51);
    [D,G] = meshgrid(dg,dgn);
    NT = zeros(size(D));
    for i = 1:numel(D)
        rBest = 1-D(i);
        prevEss = 0.75*(1-G(i));            % inverse of the degeneracy term
        NT(i) = adaptiveParticleBudget(rBest, 0, rBest, prevEss, cfg);
    end
    surf(ax,D,G,NT,'EdgeColor','none','FaceAlpha',0.95,'HandleVisibility','off');
    colormap(ax,parula); cb = colorbar(ax); cb.Label.String = 'N_k';
    hold(ax,'on');
    % Overlay where the run actually operated.
    dAct = 1 - max(prop.relGnss, prop.relLidar);
    gAct = clamp01((0.75 - [1; prop.ess(1:end-1)./prop.nParticles(1:end-1)])/0.75);
    plot3(ax,dAct,gAct,prop.nParticles+40,'.','MarkerSize',5,'Color',[0.1 0.1 0.1], ...
        'DisplayName','realized operating points');
    grid(ax,'on'); box(ax,'on'); view(ax,-40,28);
    xlabel(ax,'difficulty  1 - max(r_G,r_L)'); ylabel(ax,'degeneracy from ESS');
    zlabel(ax,'N_k');
    title(ax,'(c) adaptive budget rule  N_k = f(context, reliability, ESS)');
    legend(ax,'Location','northwest','FontSize',7);

    title(tl,'How the sensor schema and the ontology are actually applied inside the particle filter');
    exportgraphics(f,fullfile(resultDir,'ontology_pipeline.png'),'Resolution',180);
end

function drawOntologyDiagram(ax, cfg)
% Schema -> ontology -> inferred property -> filter parameter, drawn as an
% explicit data-flow so a reader can see which quantity the ontology owns.
    hold(ax,'on');
    xlim(ax,[0 100]); ylim(ax,[0 100]);
    set(ax,'XTick',[],'YTick',[],'Box','on','Color',[0.99 0.99 0.99]);
    ax.Toolbar.Visible = 'off';
    disableDefaultInteractivity(ax);

    colX = [12 37 62 87];
    w = 22; h = 15;
    hdr = {'1) Observed schema (per step k)', ...
           '2) Ontology: classes + SWRL-style rules', ...
           '3) Inferred data properties', ...
           '4) Particle-filter adaptation'};
    cCol = [0.88 0.93 0.98; 0.93 0.90 0.98; 0.90 0.96 0.90; 0.99 0.93 0.86];
    for i = 1:4
        text(ax,colX(i),97,hdr{i},'HorizontalAlignment','center', ...
            'FontWeight','bold','FontSize',8);
    end

    yr = [80 58 36 14];

    c1 = { sprintf('GnssQuality\nnumSV, HDOP,\nC/N0, hAcc'), ...
           sprintf('LidarScanQuality\nmatchScore,\nfeatureCount'), ...
           sprintf('EnvironmentContext\nOpenSky | UrbanCanyon\n| Tunnel | FeaturePoor'), ...
           sprintf('MotionControl\nspeed, yaw rate') };
    c2 = { sprintf('GnssObservation (+) context\nR1: no fix -> r_G = 0\nR2: UrbanCanyon x0.55\nR3: Tunnel -> 0   R4: OpenSky x1.08'), ...
           sprintf('LidarScanMatch (+) context\nR5: no scan -> r_L = 0\nR6: FeaturePoor x0.45\nR7: Tunnel x1.05'), ...
           sprintf('LocalizationState\nR8: r_Loc < 0.05\n-> DeadReckoningOnly'), ...
           sprintf('CovarianceFloorAxiom\nR9 / R10: never trust a\nsensor more than its floor') };
    c3 = { sprintf('gnssReliability\nr_G in [0,1]'), ...
           sprintf('lidarReliability\nr_L in [0,1]'), ...
           sprintf('localizationReliability\nr_Loc = 1-(1-r_G)(1-r_L)'), ...
           sprintf('sigma floors\n%.2f m GNSS / %.2f m LiDAR', cfg.gnssSigmaFloor, cfg.lidarPosSigmaFloor) };
    c4 = { sprintf('R_G = sigma_G^2 I_2\nsigma_G = max(floor, %.2f + %.1f(1-r_G)^2)', cfg.gnssSigmaBase, cfg.gnssSigmaGain), ...
           sprintf('R_L = diag(sigma_LP^2, sigma_LP^2, sigma_LY^2)\nsigma_LP = max(floor, %.2f + %.1f(1-r_L)^2)', cfg.lidarPosSigmaBase, cfg.lidarPosSigmaGain), ...
           sprintf('guided proposal q(x_k | x_k-1, u_k, z_k, O)\nKq = Q H'' (H Q H'' + R)^-1\nno hypothesis is deleted, only reweighted'), ...
           sprintf('particle budget\nN_k = f(1-max(r_G,r_L), ESS, r_Loc)\nN_k in [%d, %d]', cfg.Nmin, cfg.Nmax) };

    cols = {c1,c2,c3,c4};
    for j = 1:4
        for i = 1:4
            drawBox(ax,colX(j),yr(i),w,h,cols{j}{i},cCol(j,:));
        end
    end

    % schema -> ontology (rows 1..3), ontology -> property, property -> filter
    for i = 1:3
        drawArrow(ax,colX(1)+w/2,yr(i),colX(2)-w/2,yr(i));
    end
    for i = 1:4
        drawArrow(ax,colX(2)+w/2,yr(i),colX(3)-w/2,yr(i));
        drawArrow(ax,colX(3)+w/2,yr(i),colX(4)-w/2,yr(i));
    end
    % context also conditions both sensor rule blocks
    drawArrow(ax,colX(1)+w/2,yr(3)+3,colX(2)-w/2,yr(1)-h/2+2);
    drawArrow(ax,colX(1)+w/2,yr(3)+1,colX(2)-w/2,yr(2)-h/2+2);
    % r_Loc also drives the particle budget
    drawArrow(ax,colX(3)+w/2,yr(3)-4,colX(4)-w/2,yr(4)+4);
    % motion control bypasses the ontology and feeds the transition model
    yb = 4.0;
    plot(ax,[colX(1) colX(1)],[yr(4)-h/2, yb],'--','Color',[0.45 0.45 0.5], ...
        'LineWidth',0.9,'HandleVisibility','off');
    plot(ax,[colX(1) colX(4)],[yb yb],'--','Color',[0.45 0.45 0.5], ...
        'LineWidth',0.9,'HandleVisibility','off');
    drawArrow(ax,colX(4),yb,colX(4),yr(3)-h/2);
    text(ax,0.5*(colX(1)+colX(4)),yb-2.6, ...
        'odometry / IMU is shared by every variant and is NOT gated by the ontology', ...
        'HorizontalAlignment','center','FontSize',6.5,'Color',[0.35 0.35 0.4]);

    title(ax,'Schema -> ontology -> inferred reliability -> particle-filter parameters');
end

%% ========================================================================
%  FIGURE 5: WHAT THE ONTOLOGY ACTUALLY DID, STEP BY STEP
%% ========================================================================
function plotOntologyApplied(T, runs, cfg, resultDir)
    prop = runs(end).out;
    [cats, cmap] = contextColors(T);
    ctx = string(T.context);

    f = figure('Name','Ontology applied','Color','w','Position',[50 50 1560 900]);
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    % ---- (a) inferred reliability ------------------------------------
    % r_G and r_L are only meaningful at epochs where that sensor reported
    % (GNSS runs at 5 Hz on a 10 Hz timeline), so the raw values are drawn
    % as markers at those epochs plus a 2.5-s trend line.
    W = 25;
    rgv = prop.relGnss;  rgv(~T.gnss_available) = NaN;
    rlv = prop.relLidar; rlv(~T.lidar_available) = NaN;
    ax = nexttile; hold(ax,'on');
    area(ax,T.time_s,movmean(prop.relLocalization,W),'FaceColor',[0.86 0.90 0.86], ...
        'EdgeColor','none','DisplayName','r_{Loc} (trend)');
    plot(ax,T.time_s,rgv,'.','MarkerSize',3,'Color',[0.72 0.85 0.56], ...
        'HandleVisibility','off');
    plot(ax,T.time_s,rlv,'.','MarkerSize',3,'Color',[0.81 0.67 0.87], ...
        'HandleVisibility','off');
    plot(ax,T.time_s,movmean(rgv,W,'omitnan'),'-','LineWidth',1.8, ...
        'Color',[0.35 0.60 0.12],'DisplayName','r_G at GNSS epochs');
    plot(ax,T.time_s,movmean(rlv,W,'omitnan'),'-','LineWidth',1.8, ...
        'Color',[0.49 0.18 0.56],'DisplayName','r_L at LiDAR epochs');
    ylim(ax,[0 1.05]); grid(ax,'on');
    xlabel(ax,'Time [s]'); ylabel(ax,'inferred reliability');
    title(ax,'(a) ontology output: per-sensor reliability (dots = raw, line = 2.5 s trend)');
    legend(ax,'Location','southwest','FontSize',7);
    addContextBands(ax,T);

    % ---- (b) applied measurement sigma -------------------------------
    ax = nexttile; hold(ax,'on');
    plot(ax,T.time_s,prop.sigmaGnss,'.','MarkerSize',6,'Color',[0.35 0.60 0.12], ...
        'DisplayName','\sigma_G installed');
    plot(ax,T.time_s,prop.sigmaLidarPos,'.','MarkerSize',6,'Color',[0.49 0.18 0.56], ...
        'DisplayName','\sigma_{LP} installed');
    yline(ax,cfg.baseGnssSigma,'--','Color',[0.85 0.33 0.10],'LineWidth',1.3, ...
        'DisplayName',sprintf('baseline fixed \\sigma_G = %.2f m',cfg.baseGnssSigma));
    yline(ax,cfg.baseLidarPosSigma,'-.','Color',[0.85 0.33 0.10],'LineWidth',1.3, ...
        'DisplayName',sprintf('baseline fixed \\sigma_{LP} = %.2f m',cfg.baseLidarPosSigma));
    yline(ax,cfg.gnssSigmaFloor,':','Color',[0.20 0.45 0.10],'LineWidth',1.6, ...
        'DisplayName',sprintf('R9 GNSS floor %.2f m',cfg.gnssSigmaFloor));
    yline(ax,cfg.lidarPosSigmaFloor,':','Color',[0.35 0.10 0.40],'LineWidth',1.6, ...
        'DisplayName',sprintf('R10 LiDAR floor %.2f m',cfg.lidarPosSigmaFloor));
    sMax = max([prop.sigmaGnss; prop.sigmaLidarPos], [], 'omitnan');
    set(ax,'YScale','log','YTick',[0.2 0.3 0.5 0.7 0.9 1.5 2.5 4 6 10]);
    ylim(ax,[0.8*cfg.lidarPosSigmaFloor, 1.6*sMax]);
    grid(ax,'on');
    xlabel(ax,'Time [s]'); ylabel(ax,'applied \sigma [m]');
    title(ax,'(b) measurement covariance the ontology installed, per step');
    legend(ax,'Location','northwest','FontSize',6.5);
    addContextBands(ax,T,false);

    % ---- (c) rule activation raster ----------------------------------
    ax = nexttile;
    nR = numel(prop.ruleNames);
    imagesc(ax,T.time_s,1:nR,double(prop.ruleFired'));
    colormap(ax,[1 1 1; 0.16 0.38 0.68]);
    set(ax,'CLim',[0 1]);
    set(ax,'YTick',1:nR,'YTickLabel',cellstr(prop.ruleNames),'FontSize',6.5, ...
        'YDir','reverse','TickLabelInterpreter','none');
    xlabel(ax,'Time [s]');
    hold(ax,'on');
    segs = contextSegments(T);
    for i = 1:numel(segs)-1
        xline(ax,T.time_s(segs(i).i1),'-','Color',[0.85 0.35 0.10],'LineWidth',1.0);
    end
    ylim(ax,[-0.2 nR+0.5]);
    for i = 1:numel(segs)
        text(ax,0.5*(T.time_s(segs(i).i0)+T.time_s(segs(i).i1)),0.15,char(segs(i).name), ...
            'HorizontalAlignment','center','FontSize',7,'FontWeight','bold');
    end
    fired = sum(prop.ruleFired,1);
    title(ax,sprintf('(c) which ontology rule fired when  (activations: %s)', ...
        strjoin(compose('%d',fired'),'/')));

    % ---- (d) realized budget vs inferred reliability -----------------
    ax = nexttile; hold(ax,'on');
    for i = 1:numel(cats)
        m = ctx==cats(i);
        plot(ax,prop.relLocalization(m),prop.nParticles(m),'o','MarkerSize',3.5, ...
            'MarkerFaceColor',cmap(i,:),'MarkerEdgeColor','none', ...
            'DisplayName',sprintf('%s: mean N = %.0f', cats(i), mean(prop.nParticles(m),'omitnan')));
    end
    rr = linspace(0,1,101)';
    nn = arrayfun(@(x) adaptiveParticleBudget(x,0,x,1.0,cfg), rr);
    plot(ax,rr,nn,'k-','LineWidth',1.4,'DisplayName','rule, non-degenerate ESS');
    nn2 = arrayfun(@(x) adaptiveParticleBudget(x,0,x,0.30,cfg), rr);
    plot(ax,rr,nn2,'k--','LineWidth',1.2,'DisplayName','rule, ESS/N = 0.30');
    yline(ax,cfg.Nbase,'--','Color',[0.85 0.33 0.10],'LineWidth',1.2, ...
        'DisplayName','baseline fixed N');
    grid(ax,'on'); xlim(ax,[0 1.02]); ylim(ax,[0.9*cfg.Nmin 1.08*cfg.Nbase]);
    xlabel(ax,'inferred localization reliability r_{Loc}');
    ylabel(ax,'particles used N_k');
    title(ax,'(d) reliability -> sampling budget, per environment');
    legend(ax,'Location','southwest','FontSize',7);

    title(tl,'Ontology applied: inferred reliability, installed covariance, rule activations, realized budget');
    exportgraphics(f,fullfile(resultDir,'ontology_applied.png'),'Resolution',180);
end

%% ========================================================================
%  FIGURE 6: ABLATION
%% ========================================================================
function plotAblation(T, runs, ablationTbl, ctxTbl, mc, cfg, resultDir)
    names = cellstr(string({runs.name}));
    nV = numel(runs);
    hasMC = isfield(mc,'filterSeed');
    if hasMC
        S = mc.filterSeed.summary;
        rmseE = S.RMSE_std_m;
        p95E  = S.P95_std_m;
        nE    = S.MeanN_std;
        runtime = S.Runtime_median_s;
        runtimeTxt = sprintf('median over %d runs', mc.filterSeed.runs);
        srcTxt = sprintf(' (error bars: 1\\sigma over %d filter seeds)', mc.filterSeed.runs);
    else
        rmseE = []; p95E = []; nE = [];
        runtime = arrayfun(@(r) r.out.totalRuntime_s, runs)';
        runtimeTxt = 'single run, indicative only';
        srcTxt = ' (single run)';
    end
    resampRate = arrayfun(@(r) 100*mean(r.out.resampled), runs)';

    f = figure('Name','Ablation','Color','w','Position',[40 40 1700 950]);
    tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    % ---- (a) RMSE ----------------------------------------------------
    ax = nexttile;
    hb = barWithError(ax,names,ablationTbl.RMSE_m,rmseE,[0.20 0.45 0.70]);
    ylabel(ax,'Position RMSE [m]');
    title(ax,['(a) RMSE across the ablation ladder' srcTxt]);
    annotateBars(ax,hb,ablationTbl.RMSE_m,rmseE,ablationTbl.RMSE_gain_pct,'%.3f');
    ylim(ax,[0 1.34*max(ablationTbl.RMSE_m)]);

    % ---- (b) P95 -----------------------------------------------------
    ax = nexttile;
    hb = barWithError(ax,names,ablationTbl.P95_m,p95E,[0.45 0.25 0.60]);
    ylabel(ax,'P95 position error [m]');
    title(ax,'(b) 95th-percentile error (large-error suppression)');
    annotateBars(ax,hb,ablationTbl.P95_m,p95E,ablationTbl.P95_gain_pct,'%.3f');
    ylim(ax,[0 1.34*max(ablationTbl.P95_m)]);

    % ---- (c) per-context --------------------------------------------
    ax = nexttile;
    nm = cellfun(@matlab.lang.makeValidName, names, 'UniformOutput', false);
    M = zeros(height(ctxTbl), nV);
    for i = 1:nV
        M(:,i) = ctxTbl.([nm{i} '_RMSE_m']);
    end
    bar(ax, categorical(ctxTbl.Context, ctxTbl.Context), M);
    grid(ax,'on'); ylabel(ax,'RMSE [m]');
    title(ax,'(c) per-context contribution of each component');
    legend(ax,names,'Location','northwest','FontSize',7);

    % ---- (d) particle-set health -------------------------------------
    % This is where the guided proposal and the adaptive budget pay off:
    % accuracy is already saturated after PF-R, but the weight distribution
    % is not. All variants use the SAME ESS/N resampling threshold here.
    ax = nexttile;
    hb2 = bar(ax, categorical(names,names), [ablationTbl.MeanESSRatio, resampRate/100], 0.75);
    hb2(1).FaceColor = [0.25 0.55 0.35];
    hb2(2).FaceColor = [0.85 0.60 0.25];
    grid(ax,'on'); ylabel(ax,'fraction');
    if cfg.fairResampling
        thrTxt = sprintf('identical threshold ESS/N < %.2f', cfg.resampleESS);
    else
        thrTxt = sprintf('threshold %.2f / %.2f (NOT fair)', cfg.baseResampleESS, cfg.propResampleESS);
    end
    title(ax,sprintf('(d) particle-set health -- %s', thrTxt));
    legend(ax,{'mean ESS / N','share of steps resampled'},'Location','northwest','FontSize',7);
    for i = 1:nV
        text(ax,hb2(1).XEndPoints(i),ablationTbl.MeanESSRatio(i), ...
            sprintf('%.3f',ablationTbl.MeanESSRatio(i)),'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom','FontSize',7);
        text(ax,hb2(2).XEndPoints(i),resampRate(i)/100, ...
            sprintf('%.0f%%',resampRate(i)),'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom','FontSize',7);
    end
    ylim(ax,[0 0.85]);

    % ---- (e) cost ----------------------------------------------------
    ax = nexttile;
    yyaxis(ax,'left');
    hb3 = bar(ax,1:nV,ablationTbl.MeanN,0.55,'FaceColor',[0.55 0.65 0.80]);
    ylabel(ax,'mean particles used');
    ylim(ax,[0 1.25*max(ablationTbl.MeanN)]);
    for i = 1:nV
        text(ax,hb3.XEndPoints(i),ablationTbl.MeanN(i),sprintf('%.0f',ablationTbl.MeanN(i)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',7);
    end
    yyaxis(ax,'right');
    plot(ax,1:nV,runtime,'ko-','LineWidth',1.6,'MarkerFaceColor','k');
    ylabel(ax,'runtime over 300 s of data [s]');
    ylim(ax,[0.85*min(runtime) 1.20*max(runtime)]);
    for i = 1:nV
        text(ax,i,runtime(i),sprintf('  %.2f s',runtime(i)),'FontSize',7, ...
            'VerticalAlignment','bottom');
    end
    set(ax,'XTick',1:nV,'XTickLabel',names); grid(ax,'on'); xlim(ax,[0.5 nV+0.5]);
    title(ax,sprintf('(e) sampling budget and wall-clock cost (%s)',runtimeTxt));

    % ---- (f) accuracy vs cost ---------------------------------------
    ax = nexttile; hold(ax,'on');
    cmapV = lines(nV);
    voff = 0.02*range([min(ablationTbl.RMSE_m) max(ablationTbl.RMSE_m)]);
    for i = 1:nV
        plot(ax,ablationTbl.MeanN(i),ablationTbl.RMSE_m(i),'o','MarkerSize',10, ...
            'MarkerFaceColor',cmapV(i,:),'MarkerEdgeColor','k','DisplayName',names{i});
        dy = voff*(3 - 2*mod(i,2));
        text(ax,ablationTbl.MeanN(i),ablationTbl.RMSE_m(i)+dy, ...
            sprintf('%s (%.2f ms/step)',names{i},ablationTbl.StepTime_ms(i)), ...
            'FontSize',7,'HorizontalAlignment','center','VerticalAlignment','bottom');
    end
    if ~isempty(nE)
        errorbar(ax,ablationTbl.MeanN,ablationTbl.RMSE_m,rmseE,rmseE,nE,nE, ...
            'k','linestyle','none','HandleVisibility','off');
    end
    grid(ax,'on');
    xlabel(ax,'mean particles used'); ylabel(ax,'Position RMSE [m]');
    xlim(ax,[0.70*min(ablationTbl.MeanN) 1.18*max(ablationTbl.MeanN)]);
    ylim(ax,[0.97*min(ablationTbl.RMSE_m) 1.12*max(ablationTbl.RMSE_m)]);
    title(ax,'(f) accuracy vs sampling cost (lower-left is better)');
    legend(ax,'Location','east','FontSize',7);

    title(tl,'Ablation: ontology reliability -> guided proposal -> adaptive particle budget');
    exportgraphics(f,fullfile(resultDir,'ablation.png'),'Resolution',180);
end

function annotateBars(ax, hb, vals, errs, gainPct, fmt)
    for i = 1:numel(vals)
        if i == 1
            lbl = sprintf([fmt '\nbaseline'], vals(i));
        else
            lbl = sprintf([fmt '\n(%+.1f%%)'], vals(i), -gainPct(i));
        end
        text(ax,hb.XEndPoints(i),vals(i)+errAt(errs,i),lbl, ...
            'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',7);
    end
end

%% ========================================================================
%  FIGURE 7: MONTE CARLO
%% ========================================================================
function plotMonteCarlo(mc, cfg, resultDir)
    A = mc.filterSeed;
    B = mc.datasetSeed;
    names = cellstr(A.variantNames);
    nV = numel(names);
    x = 1:nV;

    f = figure('Name','Monte Carlo','Color','w','Position',[70 70 1400 860]);
    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    ax = nexttile; hold(ax,'on');
    errorbar(ax,x-0.08,A.summary.RMSE_mean_m,A.summary.RMSE_std_m,'o-', ...
        'LineWidth',1.4,'MarkerFaceColor','auto', ...
        'DisplayName',sprintf('A: fixed dataset, %d filter seeds',A.runs));
    errorbar(ax,x+0.08,B.summary.RMSE_mean_m,B.summary.RMSE_std_m,'s--', ...
        'LineWidth',1.4,'MarkerFaceColor','auto', ...
        'DisplayName',sprintf('B: %d dataset + filter seeds',B.runs));
    set(ax,'XTick',x,'XTickLabel',names); grid(ax,'on'); xlim(ax,[0.6 nV+0.4]);
    ylabel(ax,'Position RMSE [m]  (mean \pm 1\sigma)');
    title(ax,'(a) RMSE with Monte-Carlo uncertainty');
    legend(ax,'Location','northeast','FontSize',7);

    ax = nexttile; hold(ax,'on');
    errorbar(ax,x-0.08,A.summary.P95_mean_m,A.summary.P95_std_m,'o-','LineWidth',1.4, ...
        'DisplayName','A: filter seed');
    errorbar(ax,x+0.08,B.summary.P95_mean_m,B.summary.P95_std_m,'s--','LineWidth',1.4, ...
        'DisplayName','B: dataset + filter seed');
    set(ax,'XTick',x,'XTickLabel',names); grid(ax,'on'); xlim(ax,[0.6 nV+0.4]);
    ylabel(ax,'P95 error [m]  (mean \pm 1\sigma)');
    title(ax,'(b) P95 error with Monte-Carlo uncertainty');
    legend(ax,'Location','northeast','FontSize',7);

    ax = nexttile; hold(ax,'on');
    relA = 100*(A.rmse(:,1)-A.rmse(:,end))./A.rmse(:,1);
    relB = 100*(B.rmse(:,1)-B.rmse(:,end))./B.rmse(:,1);
    edges = linspace(min([relA;relB;0])-2, max([relA;relB])+2, 22);
    histogram(ax,relA,edges,'FaceColor',[0.20 0.45 0.70],'EdgeColor','none', ...
        'DisplayName','A: filter seed');
    histogram(ax,relB,edges,'FaceColor',[0.90 0.55 0.15],'EdgeColor','none', ...
        'DisplayName','B: dataset + filter seed');
    xline(ax,0,'k-','LineWidth',1.2,'HandleVisibility','off');
    xline(ax,mean(relA),'--','Color',[0.20 0.45 0.70],'LineWidth',1.3, ...
        'Label',sprintf('mean %.1f%%',mean(relA)),'FontSize',7,'HandleVisibility','off');
    xline(ax,mean(relB),'--','Color',[0.90 0.55 0.15],'LineWidth',1.3, ...
        'Label',sprintf('mean %.1f%%',mean(relB)),'FontSize',7,'HandleVisibility','off');
    grid(ax,'on');
    xlabel(ax,sprintf('paired RMSE reduction %s -> %s [%%]',names{1},names{end}));
    ylabel(ax,'runs');
    title(ax,sprintf('(c) paired per-run improvement (wins %d/%d and %d/%d)', ...
        sum(relA>0),numel(relA),sum(relB>0),numel(relB)));
    legend(ax,'Location','northwest','FontSize',7);

    ax = nexttile;
    yyaxis(ax,'left');
    hb = bar(ax,x,A.summary.Runtime_median_s,0.55,'FaceColor',[0.55 0.65 0.80]);
    ylabel(ax,'median runtime over 300 s of data [s]');
    ylim(ax,[0 1.35*max(A.summary.Runtime_median_s)]);
    for i = 1:nV
        text(ax,hb.XEndPoints(i),A.summary.Runtime_median_s(i), ...
            sprintf('%.2f s',A.summary.Runtime_median_s(i)),'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom','FontSize',7);
    end
    yyaxis(ax,'right');
    plot(ax,x,A.summary.MeanN_mean,'ko-','LineWidth',1.5,'MarkerFaceColor','k');
    ylabel(ax,'mean particles used');
    ylim(ax,[0 1.25*max(A.summary.MeanN_mean)]);
    set(ax,'XTick',x,'XTickLabel',names); grid(ax,'on'); xlim(ax,[0.5 nV+0.5]);
    title(ax,sprintf('(d) median runtime over %d runs vs sampling budget',A.runs));

    title(tl,sprintf('Monte-Carlo repetition (%d runs per mode); runtime reported as a median', A.runs));
    exportgraphics(f,fullfile(resultDir,'monte_carlo.png'),'Resolution',180);
end

function v = errAt(errs, i)
    if isempty(errs) || ~isfinite(errs(i))
        v = 0;
    else
        v = errs(i);
    end
end
