function cfg = default_config()
%DEFAULT_CONFIG Experiment configuration.

cfg.seed = 42;

% Data
cfg.dataDir = fullfile(pwd,'data');
cfg.files.middle = 'medium.csv';
cfg.files.harsh  = 'harsh.csv';
cfg.files.deep   = 'deep.csv';

cfg.features = { ...
    'numSV','hDOP','vDOP','hAcc','vAcc','gSpeed', ...
    'CN0_mean','CN0_std','CN0_gap','low_elev_ratio', ...
    'PR_RMS','Fault_SVID_count'};

cfg.faultThresholdM = 3.0;
cfg.softLabelTemperature = 1.0;

% Use the dataset's soft_fault_prob_surrogate column as the weak label.
% Off: weak labels come from a sigmoid on the GT 2D error (see load_gnss_csv).
cfg.useSurrogateSoftLabel = false;

% Sliding window
cfg.window = 30;
cfg.stride = 1;

% Training
cfg.training.epochs = 25;
cfg.training.batchSize = 32;
cfg.training.learnRate = 1e-3;
cfg.training.gradClip = 5.0;
cfg.training.verboseEvery = 25;

% Prior-study Transformer settings from the provided presentation
cfg.transformer.dModel = 24;
cfg.transformer.numHeads = 4;
cfg.transformer.numLayers = 2;
cfg.transformer.ffnDim = 64;

% Proposed TA-GAT
cfg.tagat.nodeStatDim = 5;  % mean,std,last,delta,slope
cfg.tagat.hiddenDim = 24;
cfg.tagat.numLayers = 2;
cfg.tagat.leakySlope = 0.2;

% Temporal-adaptive graph
cfg.graph.alphaOnt = 0.40;
cfg.graph.alphaCov = 0.30;
cfg.graph.alphaDyn = 0.30;
cfg.graph.edgeFloor = 1e-4;
cfg.graph.priorBias = 0.75;

% Dual EDL
cfg.loss.betaWeak = 0.5;
cfg.loss.kldWeightHard = 1e-3;
cfg.loss.kldWeightWeakMax = 1e-3;
cfg.loss.annealEpochs = 10;

% EDL KL uses gammaln/psi. If a local MATLAB release does not support
% automatic differentiation for psi, set false and rerun.
cfg.loss.useKLD = true;

cfg.resultsDir = fullfile(pwd,'results');
end
