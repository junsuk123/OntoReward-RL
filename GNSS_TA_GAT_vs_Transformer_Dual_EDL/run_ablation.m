clear; clc; close all;

cfg0=default_config();
rng(cfg0.seed);
if ~exist(cfg0.resultsDir,'dir'), mkdir(cfg0.resultsDir); end

Dmid=load_gnss_csv(fullfile(cfg0.dataDir,cfg0.files.middle),cfg0,'Middle');
Dhar=load_gnss_csv(fullfile(cfg0.dataDir,cfg0.files.harsh),cfg0,'Harsh');
Ddeep=load_gnss_csv(fullfile(cfg0.dataDir,cfg0.files.deep),cfg0,'Deep');

tr={Dmid,Dhar}; te={Ddeep};
[mu,sigma]=fit_standardizer(tr);
tr=apply_standardizer(tr,mu,sigma);
te=apply_standardizer(te,mu,sigma);

[Xtr,YtrH,YtrS]=build_windows(tr,cfg0);
[Xte,YteH,~]=build_windows(te,cfg0);

% 1) Prior-study baseline
fprintf('\n[1/4] Transformer-Dual EDL\n');
b=train_transformer_dual_edl(Xtr,YtrH,YtrS,cfg0);
[p,~]=predict_transformer_dual_edl(b,Xte,cfg0);
M(1)=binary_metrics(YteH,p>=0.5); %#ok<SAGROW>

Aont=build_ontology_adjacency(cfg0.features);
C=make_ablation_configs();

names=["Transformer-Dual EDL", ...
       "Static Ontology-GAT", ...
       "Dynamic GAT (No Ontology)", ...
       "Full Ontology TA-GAT"];

cfgs={C.staticOntology,C.dynamicNoOntology,C.fullTAGAT};

for q=1:3
    cfg=cfgs{q};
    fprintf('\n[%d/4] %s\n',q+1,names(q+1));
    [Str,Atr]=build_temporal_graph_inputs(Xtr,Aont,cfg);
    [Ste,Ate]=build_temporal_graph_inputs(Xte,Aont,cfg);
    g=train_tagat_dual_edl(Str,Atr,YtrH,YtrS,cfg);
    [p,~]=predict_tagat_dual_edl(g,Ste,Ate,cfg);
    M(q+1)=binary_metrics(YteH,p>=0.5); %#ok<SAGROW>
end

Precision=arrayfun(@(x)x.precision,M).';
Recall=arrayfun(@(x)x.recall,M).';
F1=arrayfun(@(x)x.f1,M).';
Accuracy=arrayfun(@(x)x.accuracy,M).';
FPR=arrayfun(@(x)x.fpr,M).';

T=table(names.',Precision,Recall,F1,Accuracy,FPR, ...
    'VariableNames',{'Model','Precision','Recall','F1','Accuracy','FPR'});
disp(T);
writetable(T,fullfile(cfg0.resultsDir,'ablation_metrics.csv'));
save(fullfile(cfg0.resultsDir,'ablation_result.mat'),'T','M','mu','sigma','cfg0');

fprintf('\nAblation results saved.\n');
