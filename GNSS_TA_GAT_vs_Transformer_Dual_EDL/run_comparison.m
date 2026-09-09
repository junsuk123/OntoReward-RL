clear; clc; close all;

cfg = default_config();
rng(cfg.seed);

if ~exist(cfg.resultsDir,'dir')
    mkdir(cfg.resultsDir);
end

fprintf('\n=== GNSS Fault Detection Comparison ===\n');
fprintf('Baseline : Transformer-Dual EDL\n');
fprintf('Proposed : Ontology-Guided Temporal-Adaptive GAT + Dual EDL\n\n');

% 1) Load
Dmid  = load_gnss_csv(fullfile(cfg.dataDir,cfg.files.middle),cfg,'Middle');
Dhar  = load_gnss_csv(fullfile(cfg.dataDir,cfg.files.harsh), cfg,'Harsh');
Ddeep = load_gnss_csv(fullfile(cfg.dataDir,cfg.files.deep),  cfg,'Deep');

% Concatenate train trajectories only AFTER keeping trajectory boundaries.
trainSeq = {Dmid, Dhar};
testSeq  = {Ddeep};

% 2) Standardization from training only
[mu,sigma] = fit_standardizer(trainSeq);
trainSeq = apply_standardizer(trainSeq,mu,sigma);
testSeq  = apply_standardizer(testSeq,mu,sigma);

% 3) Sliding windows
[Xtr,YtrHard,YtrSoft,metaTr] = build_windows(trainSeq,cfg);
[Xte,YteHard,YteSoft,metaTe] = build_windows(testSeq,cfg);

fprintf('Train windows: %d\n', size(Xtr,3));
fprintf('Test  windows: %d\n', size(Xte,3));
fprintf('Train fault ratio: %.3f\n', mean(YtrHard));
fprintf('Test  fault ratio: %.3f\n\n', mean(YteHard));

% 4) Baseline
fprintf('--- Train baseline Transformer-Dual EDL ---\n');
baseline = train_transformer_dual_edl(Xtr,YtrHard,YtrSoft,cfg);

fprintf('\n--- Evaluate baseline ---\n');
[pBase,uBase] = predict_transformer_dual_edl(baseline,Xte,cfg);
mBase = binary_metrics(YteHard,pBase >= 0.5);

% 5) Proposed
fprintf('\n--- Build temporal-adaptive graph tensors ---\n');
Aont = build_ontology_adjacency(cfg.features);
[StatsTr, Atr] = build_temporal_graph_inputs(Xtr,Aont,cfg);
[StatsTe, Ate] = build_temporal_graph_inputs(Xte,Aont,cfg);

fprintf('--- Train proposed Ontology-Guided TA-GAT ---\n');
tagat = train_tagat_dual_edl(StatsTr,Atr,YtrHard,YtrSoft,cfg);

fprintf('\n--- Evaluate proposed ---\n');
[pTAGAT,uTAGAT] = predict_tagat_dual_edl(tagat,StatsTe,Ate,cfg);
mTAGAT = binary_metrics(YteHard,pTAGAT >= 0.5);

% 6) Save metrics
Model = ["Transformer-Dual EDL"; "Ontology-TA-GAT + Dual EDL"];
Precision = [mBase.precision; mTAGAT.precision];
Recall = [mBase.recall; mTAGAT.recall];
F1 = [mBase.f1; mTAGAT.f1];
Accuracy = [mBase.accuracy; mTAGAT.accuracy];
FPR = [mBase.fpr; mTAGAT.fpr];

T = table(Model,Precision,Recall,F1,Accuracy,FPR);
disp(T);
writetable(T,fullfile(cfg.resultsDir,'comparison_metrics.csv'));

save(fullfile(cfg.resultsDir,'comparison_result.mat'), ...
    'cfg','baseline','tagat','mu','sigma','Aont', ...
    'pBase','uBase','pTAGAT','uTAGAT','YteHard','YteSoft','metaTe','T');

plot_comparison(YteHard,pBase,pTAGAT,uBase,uTAGAT,cfg);

fprintf('\nSaved to: %s\n', cfg.resultsDir);
