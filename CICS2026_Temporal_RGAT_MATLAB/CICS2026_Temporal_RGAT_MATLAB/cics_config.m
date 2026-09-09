function cfg = cics_config(mode)
%CICS_CONFIG Central configuration for CICS2026 benchmark.
if nargin<1, mode="demo"; end
mode=string(mode);

cfg.root = fileparts(mfilename('fullpath'));
cfg.mode = mode;
% Only the explicitly non-strict mode may fall back to a surrogate soft label.
cfg.allowSurrogateSoftLabel = false;
cfg.seed = 260818;
cfg.featureNames = ["numSV","hDOP","vDOP","hAcc","vAcc","gSpeed", ...
    "CNO_mean","CNO_std","CNO_gap","low_elev_ratio","PR_RMS","Fault_SVID_count"];
cfg.priorWeights = single([1.0 1.5 1.5 2.0 2.0 1.0 1.0 1.5 1.5 2.0 1.5 1.5]);
cfg.window = 30;
cfg.threshold = 0.5;
cfg.hardFaultThresholdM = 3.0;

% Prior Transformer
cfg.transformer.dModel = 24;
cfg.transformer.numHeads = 4;
cfg.transformer.ffn = 64;
cfg.transformer.numLayers = 2;
cfg.transformer.dropout = 0.10;

% Temporal R-GAT
cfg.rgat.hidden = 24;
cfg.rgat.numHeads = 4;
cfg.rgat.numLayers = 2;
cfg.rgat.maxLag = 5;      % 0..5 past steps jointly considered
cfg.rgat.timeDim = 6;     % sinusoidal lag encoding dimension
cfg.rgat.useTimeEncoding = true;

% EDL
cfg.edl.betaWeak = 1.0;
cfg.edl.annealEpochs = 10;
cfg.edl.klHardWeight = 1.0;
cfg.edl.klWeakWeight = 1.0;

% Training
cfg.train.learnRate = 1e-3;
cfg.train.miniBatchSize = 32;
cfg.train.gradDecay = 0.9;
cfg.train.sqGradDecay = 0.999;
cfg.train.gradientClip = 5.0;

% ---- Model selection -------------------------------------------------------
% Windows overlap by W-1 samples, so a network can drive training loss toward
% zero by memorising them. Held-out selection is what makes the reported number
% mean anything; without it the last epoch is reported no matter how bad it is.
cfg.train.patience = 8;

% ---- Split protocol --------------------------------------------------------
% "pooled_chronological": every scenario contributes a chronological
%   train/val/test block. Temporal order is preserved inside each scenario.
% "cross_scenario": the original protocol - fit medium+harsh, test deep.
%   Measured leave-one-scenario-out AUROC on this dataset: medium 0.334,
%   harsh 0.439, deep 0.457. All three are at or BELOW chance, because the
%   sign of the feature-fault relationship flips between environments
%   (gSpeed correlates +0.33 with faults in medium and -0.15 in deep).
%   Under that protocol no architecture can score above chance, so it cannot
%   be used to compare architectures. Kept because it is a legitimate
%   generalisation question with a legitimate - negative - answer.
cfg.split.mode = "pooled_chronological";
cfg.split.trainFrac = 0.60;
cfg.split.valFrac = 0.15;
% Purge this many windows on each side of a chronological boundary so that no
% evaluation window shares a raw epoch with a training window.
cfg.split.purgeWindows = cfg.window;

% ---- Weak (soft) label source ---------------------------------------------
% Non-strict mode only. "pr_rms" is the documented PR_RMS surrogate shipped with
% the public reconstruction. Diagnostic: as a direct predictor of hard_label it
% scores AUROC 0.471 - below chance - so the weak EDL head trained on it learns
% noise and the dual fusion is strictly worse than the hard head alone.
% "pos_error" is a smooth logistic of pos_error_2d around the same 3 m threshold
% that defines hard_label. It reproduces the prior study's evident design (their
% SoftLabel baseline alone scored F1 0.959, so their soft label is error-derived)
% but it is CIRCULAR: it cannot be used to claim the weak head adds information.
cfg.weakLabel.source = "pos_error";
cfg.weakLabel.errorTauM = 1.0;

if mode=="demo"
    cfg.train.epochs = 2;
    cfg.demoMaxTrainWindows = 512;
    cfg.demoMaxTestWindows = 300;
    cfg.rgat.numHeads = 2;
    cfg.rgat.numLayers = 1;
    cfg.rgat.maxLag = 2;
    cfg.runAblations = false;
    cfg.runForecast = false;
    cfg.bootstrapSamples = 300;
    cfg.dataDir = fullfile(cfg.root,'data','demo');
    cfg.outputDir = fullfile(cfg.root,'outputs','DEMO_SMOKE_TEST_ONLY');
else
    % Validation-selected epochs peak early on this data (typically 3-10), so this
    % cap bounds wall-clock rather than defining training length. Early stopping
    % via cfg.train.patience is what actually ends a run.
    cfg.train.epochs = 30;
    cfg.runAblations = true;
    cfg.runForecast = true;
    cfg.forecastHorizons = [1 5 10];
    cfg.bootstrapSamples = 2000;
    stamp=char(datetime('now','Format','yyyyMMdd_HHmmss'));
    if mode=="nonstrict"
        % Same architecture, split, epochs and statistics as strict real mode, but
        % reading the public reconstruction, whose soft label is a documented
        % surrogate rather than the prior study's estimator. Separate mode so a
        % strict run is unaffected, and so every artefact of a surrogate run lands
        % under a directory whose name says what it is.
        cfg.dataDir = fullfile(cfg.root,'data','urbannav_public');
        cfg.outputDir = fullfile(cfg.root,'outputs',['NONSTRICT_SURROGATE_' stamp]);
        cfg.allowSurrogateSoftLabel = true;
    else
        cfg.dataDir = fullfile(cfg.root,'data','real');
        cfg.outputDir = fullfile(cfg.root,'outputs',['REAL_' stamp]);
    end
end

% Raw public UrbanNav-HK inputs for build_urbannav_features. Scenario folders
% hold the receiver logs under gnss/ plus the raw ground-truth trajectory.
cfg.rawDataDir = fullfile(cfg.root,'data','raw');
cfg.urbannav.medium.datasetName = "UrbanNav-HK-Medium-Urban-1";
cfg.urbannav.harsh.datasetName  = "UrbanNav-HK-Harsh-Urban-1";
cfg.urbannav.deep.datasetName   = "UrbanNav-HK-Deep-Urban-1";
cfg.urbannav.medium.gtName = "UrbanNav_TST_GT_raw.txt";
cfg.urbannav.harsh.gtName  = "UrbanNav_mongkok_GT_part_raw.txt";
cfg.urbannav.deep.gtName   = "UrbanNav_whampoa_raw.txt";
% Epoch/fault counts quoted in the prior presentation, for cross-checking only.
cfg.urbannav.slideDiagnostics.medium = [658 206];
cfg.urbannav.slideDiagnostics.harsh  = [2314 1569];
cfg.urbannav.slideDiagnostics.deep   = [1539 521];

cfg.files.medium = fullfile(cfg.dataDir,'medium.csv');
cfg.files.harsh  = fullfile(cfg.dataDir,'harsh.csv');
cfg.files.deep   = fullfile(cfg.dataDir,'deep.csv');

% Baseline reproduction gate from prior presentation.
cfg.reproduction.targetF1 = 0.950;
cfg.reproduction.tolerance = 0.020;

cfg.modelList = [ ...
    "Transformer", ...
    "TransEDL-Hard", ...
    "TransEDL-Soft", ...
    "SoftLabel", ...
    "Transformer-DualEDL", ...
    "R-GAT-DualEDL", ...
    "R-GAT+Transformer-DualEDL", ...
    "TemporalR-GAT-DualEDL"];
if cfg.runAblations
    cfg.modelList=[cfg.modelList, ...
        "TemporalR-GAT-NoTimeEncoding", ...
        "TemporalR-GAT-NoRelationType", ...
        "TemporalR-GAT-1Head", ...
        "TemporalR-GAT-1Layer"];
end
end
