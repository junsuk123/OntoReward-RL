function export_benchmark_results(results,boot,mc,baseGate,metaTest,forecastTable,mu,sigma,cfg)
%EXPORT_BENCHMARK_RESULTS Write all metrics, predictions, statistics, manifest.
keys=fieldnames(results); rows={};
for i=1:numel(keys)
    R=results.(keys{i}); M=R.metrics;
    rows(end+1,:)={char(R.name),M.TP,M.TN,M.FP,M.FN,M.Accuracy,M.Precision,M.Recall,M.Specificity, ...
        M.F1,M.AUROC,M.AUPRC,M.Brier,M.NLL,M.ECE,M.MeanUncertainty,M.MeanUncertaintyCorrect,M.MeanUncertaintyIncorrect,thr_of(R)}; %#ok<AGROW>
    P=metaTest; P.hard_label=R.Yhard(:); P.soft_fault_prob=R.Ysoft(:); P.p_fault=R.pFault(:); P.uncertainty=R.uncertainty(:);
    writetable(P,fullfile(cfg.outputDir,['predictions_' keys{i} '.csv']));
end
metrics=cell2table(rows,'VariableNames',{'Model','TP','TN','FP','FN','Accuracy','Precision','Recall','Specificity','F1','AUROC','AUPRC','Brier','NLL','ECE','MeanUncertainty','MeanUncertaintyCorrect','MeanUncertaintyIncorrect','Threshold'});
writetable(metrics,fullfile(cfg.outputDir,'metrics.csv'));
% Compare reproduced metrics with values exactly as displayed in the prior presentation.
priorFile=fullfile(cfg.root,'prior_reported_metrics.csv');
if isfile(priorFile)
    P=readtable(priorFile);
    C=outerjoin(P,metrics(:,{'Model','Precision','Recall','F1'}),'Keys','Model','MergeKeys',true,'Type','left');
    C.DeltaPrecision=C.Precision-C.ReportedPrecision;
    C.DeltaRecall=C.Recall-C.ReportedRecall;
    C.DeltaF1=C.F1-C.ReportedF1;
    writetable(C,fullfile(cfg.outputDir,'prior_reported_comparison.csv'));
end
writetable(struct2table(boot),fullfile(cfg.outputDir,'paired_bootstrap_vs_prior.csv'));
writetable(struct2table(mc),fullfile(cfg.outputDir,'mcnemar_vs_prior.csv'));
if ~isempty(forecastTable), writetable(forecastTable,fullfile(cfg.outputDir,'forecast_metrics.csv')); end
manifest=struct(); manifest.mode=char(cfg.mode); manifest.created=char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')); manifest.seed=cfg.seed;
manifest.splitMode=char(cfg.split.mode);
manifest.splitTrainFrac=cfg.split.trainFrac; manifest.splitValFrac=cfg.split.valFrac;
manifest.splitPurgeWindows=cfg.split.purgeWindows;
if cfg.split.mode=="cross_scenario"
    manifest.trainScenarios={'medium','harsh'}; manifest.testScenario='deep';
else
    manifest.trainScenarios={'medium','harsh','deep'}; manifest.testScenario='medium+harsh+deep (chronological tail)';
end
% cfg.weakLabel.source only applies when the surrogate path is taken. In strict
% and demo modes the weak label is the file's own soft_fault_prob column, and
% naming a surrogate here would misdescribe the supervision actually used.
if cfg.allowSurrogateSoftLabel
    manifest.weakLabelSource=char(cfg.weakLabel.source);
else
    manifest.weakLabelSource='file:soft_fault_prob';
end
manifest.thresholdSelection='max-F1 on validation split';
manifest.window=cfg.window;
manifest.featureNames=cellstr(cfg.featureNames); manifest.normalizationMean=double(mu); manifest.normalizationStd=double(sigma);
manifest.priorF1Target=cfg.reproduction.targetF1; manifest.priorTolerance=cfg.reproduction.tolerance;
manifest.baselineReproductionPass=logical(baseGate); manifest.note='Demo metrics are smoke-test only; real claims require strict processed UrbanNav data.';
manifest.strictReproduction=cfg.mode=="real";
manifest.surrogateSoftLabel=logical(cfg.allowSurrogateSoftLabel);
if cfg.allowSurrogateSoftLabel
    manifest.note=['NON-STRICT: the weak head was trained on soft_fault_prob_surrogate, ' ...
        'a logistic function of PR_RMS, not the prior study''s residual-to-probability ' ...
        'estimator. Dual-EDL metrics here are not comparable with the reported F1=0.950 ' ...
        'baseline and the reproduction gate is not interpretable.'];
    write_nonstrict_notice(cfg);
end
fid=fopen(fullfile(cfg.outputDir,'run_manifest.json'),'w'); fprintf(fid,'%s',jsonencode(manifest,PrettyPrint=true)); fclose(fid);
end

function write_nonstrict_notice(cfg)
%WRITE_NONSTRICT_NOTICE Drop a plain-language warning beside the CSVs, so the
% directory cannot be read as a reproduction even out of context.
fid=fopen(fullfile(cfg.outputDir,'NOT_A_REPRODUCTION.md'),'w');
fprintf(fid,'# Not a reproduction of the prior study\n\n');
fprintf(fid,'Every metric in this directory was produced with `cics_config("nonstrict")`.\n\n');
fprintf(fid,'## What is real\n\n');
fprintf(fid,'- The 12 features and `hard_label`, rebuilt from the official public UrbanNav\n');
fprintf(fid,'  u-blox F9P NMEA stream and raw ground truth by `build_urbannav_features`.\n');
fprintf(fid,'- The train Medium+Harsh / test Deep Urban split, window length and 3 m hard\n');
fprintf(fid,'  fault threshold.\n');
fprintf(fid,'- The architectures, ablations, forecast horizons and statistical tests.\n\n');
fprintf(fid,'## What is not\n\n');
fprintf(fid,'The weak supervision. `soft_fault_prob` does not exist in these tables, so the\n');
fprintf(fid,'Dual-EDL weak head was trained on `soft_fault_prob_surrogate`, a logistic\n');
fprintf(fid,'function of `PR_RMS`. The prior presentation never defines its\n');
fprintf(fid,'residual-to-probability mapping, so the real one could not be used.\n\n');
fprintf(fid,'## Consequences\n\n');
fprintf(fid,'- Do NOT compare any number here against the reported F1=0.950 baseline.\n');
fprintf(fid,'- `baselineReproductionPass` in the manifest is not interpretable in this mode.\n');
fprintf(fid,'- Comparisons BETWEEN models in this directory are meaningful: every model saw\n');
fprintf(fid,'  identical data, split and supervision. That makes it valid for architecture\n');
fprintf(fid,'  and ablation conclusions, and invalid for reproduction claims.\n\n');
fprintf(fid,'For a paper-comparable run, supply the prior study''s processed tables with a\n');
fprintf(fid,'genuine `soft_fault_prob` in `data/real/` and use `run_all_real("strict")`.\n');
fclose(fid);
end

function t=thr_of(R)
%THR_OF Validation-selected decision threshold for one result, 0.5 if absent.
if isfield(R,'threshold') && ~isempty(R.threshold), t=double(R.threshold); else, t=0.5; end
end
