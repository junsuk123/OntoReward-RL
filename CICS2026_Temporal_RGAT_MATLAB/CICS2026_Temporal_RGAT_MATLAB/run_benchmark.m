function run_benchmark(mode)
%RUN_BENCHMARK Full prior-study comparison and proposed-model validation.
cfg=cics_config(mode); rng(cfg.seed,'twister');
check_prior_reported_consistency();
if ~isfolder(cfg.outputDir), mkdir(cfg.outputDir); end
modelDir=fullfile(cfg.outputDir,'models'); if ~isfolder(modelDir), mkdir(modelDir); end
D=load_scenarios(cfg); [D,mu,sigma]=normalize_scenarios(D,cfg);
report_label_quality(D,cfg);

% Prior study manually weighted 12 features for Transformer baselines.
Dbase=D;
for s=["medium","harsh","deep"]
    Dbase.(char(s)).X=D.(char(s)).X.*cfg.priorWeights;
end
SP=make_splits(D,cfg,0);        % proposed-model feature scaling
SPb=make_splits(Dbase,cfg,0);   % prior-weighted feature scaling
assert(isequal(SP.YhTrain,SPb.YhTrain) && isequal(SP.YhTest,SPb.YhTest));
describe_splits(SP);
metaTest=SP.metaTest;

% Keep demo smoke test fast while preserving the exact full-data real mode.
if cfg.mode=="demo"
    [SP,SPb,metaTest]=subsample_demo(SP,SPb,cfg);
end

val=struct('X',SP.Xval,'Yh',SP.YhVal,'Ys',SP.YsVal);
valb=struct('X',SPb.Xval,'Yh',SPb.YhVal,'Ys',SPb.YsVal);

results=struct(); histories=struct(); trained=struct();
for name=cfg.modelList
    key=matlab.lang.makeValidName(char(name));
    if name=="SoftLabel"
        % Tuned on validation like every trained model, so the comparison is fair.
        sThr=select_threshold(SP.YhVal,SP.YsVal,cfg.threshold);
        Msoft=compute_metrics(SP.YhTest,SP.YsTest,sThr); Msoft.Threshold=sThr;
        Msoft.MeanUncertainty=NaN; Msoft.MeanUncertaintyCorrect=NaN; Msoft.MeanUncertaintyIncorrect=NaN;
        R=struct('name',name,'pFault',SP.YsTest,'uncertainty',nan(size(SP.YsTest),'single'), ...
            "metrics",Msoft,"Yhard",SP.YhTest,"Ysoft",SP.YsTest,"threshold",sThr);
        results.(key)=R; continue
    end
    model=build_model(name,cfg);
    if startsWith(name,"Transformer") || startsWith(name,"TransEDL")
        S=SPb; V=valb;
    else
        S=SP; V=val;
    end
    [model,hist]=train_model(model,S.Xtrain,S.YhTrain,S.YsTrain,cfg,V);
    R=evaluate_model(model,S.Xtest,S.YhTest,S.YsTest,cfg);
    fprintf('  test | F1 %.4f  AUROC %.4f  Acc %.4f\n',R.metrics.F1,R.metrics.AUROC,R.metrics.Accuracy);
    results.(key)=R; histories.(key)=hist; trained.(key)=model;
    writetable(hist,fullfile(cfg.outputDir,['training_history_' key '.csv']));
    save(fullfile(modelDir,[key '.mat']),'model','cfg','-v7.3');
end

priorKey=matlab.lang.makeValidName('Transformer-DualEDL'); newKey=matlab.lang.makeValidName('TemporalR-GAT-DualEDL');
baseGate=abs(results.(priorKey).metrics.F1-cfg.reproduction.targetF1)<=cfg.reproduction.tolerance;
if cfg.mode=="real" && ~baseGate
    warning(['Prior Transformer-DualEDL baseline failed reproduction gate. ' ...
        'Do not claim proposed-model improvement until preprocessing/training alignment is resolved.']);
elseif cfg.mode=="nonstrict"
    % The gate compares against a baseline trained on the prior soft label, which
    % this mode does not have, so its outcome carries no information either way.
    warning(['Non-strict mode: the reproduction gate is not interpretable because the ' ...
        'weak head was trained on a surrogate soft label. Treat these metrics as an ' ...
        'architecture/ablation comparison only.']);
end

thrPair=[results.(priorKey).threshold results.(newKey).threshold];
boot=paired_bootstrap_compare(SP.YhTest,results.(priorKey).pFault,results.(newKey).pFault, ...
    thrPair,cfg.bootstrapSamples,cfg.seed+1);
mc=mcnemar_exact(SP.YhTest,results.(priorKey).pFault,results.(newKey).pFault,thrPair);

% Attention diagnostics on one high-fault-probability test window.
if isfield(trained,newKey)
    [~,ix]=max(results.(newKey).pFault);
    export_attention(trained.(newKey),SP.Xtest(:,:,ix),cfg);
end

forecastTable=table();
if cfg.runForecast
    forecastTable=run_forecast_benchmark(D,cfg);
end
export_benchmark_results(results,boot,mc,baseGate,metaTest,forecastTable,mu,sigma,cfg);
plot_benchmark_results(results,SP.YhTest,cfg);

fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Split protocol: %s\n',SP.mode);
fprintf('Weak label source: %s\n',cfg.weakLabel.source);
fprintf('Prior Transformer-DualEDL F1: %.4f (AUROC %.4f)\n', ...
    results.(priorKey).metrics.F1,results.(priorKey).metrics.AUROC);
fprintf('Proposed Temporal R-GAT-DualEDL F1: %.4f (AUROC %.4f)\n', ...
    results.(newKey).metrics.F1,results.(newKey).metrics.AUROC);
fprintf('Delta F1: %.4f | bootstrap 95%% CI [%.4f, %.4f] | p=%.4g\n', ...
    boot.MeanDeltaF1,boot.CI95Low,boot.CI95High,boot.Pbootstrap);
fprintf('Baseline reproduction gate: %d\n',baseGate);
fprintf('Outputs: %s\n',cfg.outputDir);
end

function [SP,SPb,metaTest]=subsample_demo(SP,SPb,cfg)
%SUBSAMPLE_DEMO Thin the smoke test without changing the split semantics.
it=unique(round(linspace(1,size(SP.Xtrain,3),min(cfg.demoMaxTrainWindows,size(SP.Xtrain,3)))));
ie=unique(round(linspace(1,size(SP.Xtest,3),min(cfg.demoMaxTestWindows,size(SP.Xtest,3)))));
iv=unique(round(linspace(1,size(SP.Xval,3),min(cfg.demoMaxTestWindows,size(SP.Xval,3)))));
for f=["Xtrain","Xval","Xtest"]
    switch f
        case "Xtrain", ii=it; case "Xval", ii=iv; otherwise, ii=ie;
    end
    SP.(char(f))=SP.(char(f))(:,:,ii); SPb.(char(f))=SPb.(char(f))(:,:,ii);
end
SP.YhTrain=SP.YhTrain(it); SP.YsTrain=SP.YsTrain(it);
SP.YhVal=SP.YhVal(iv);     SP.YsVal=SP.YsVal(iv);
SP.YhTest=SP.YhTest(ie);   SP.YsTest=SP.YsTest(ie);
SPb.YhTrain=SP.YhTrain; SPb.YsTrain=SP.YsTrain;
SPb.YhVal=SP.YhVal;     SPb.YsVal=SP.YsVal;
SPb.YhTest=SP.YhTest;   SPb.YsTest=SP.YsTest;
SP.metaTest=SP.metaTest(ie,:); metaTest=SP.metaTest;
end
