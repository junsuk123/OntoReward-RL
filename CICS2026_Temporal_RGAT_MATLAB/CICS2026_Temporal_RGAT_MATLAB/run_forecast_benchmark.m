function T = run_forecast_benchmark(D,cfg)
%RUN_FORECAST_BENCHMARK Compare future fault prediction at multiple horizons.
% Uses the same split protocol, purge and validation-based epoch selection as
% the main benchmark, so horizons are comparable with the k=0 row.
rows={};
Dbase=D; for s=["medium","harsh","deep"], Dbase.(char(s)).X=D.(char(s)).X.*cfg.priorWeights; end
for h=cfg.forecastHorizons
    fprintf('\n=== Forecast horizon k=%d ===\n',h);
    SP=make_splits(D,cfg,h); SPb=make_splits(Dbase,cfg,h);
    assert(isequal(SP.YhTrain,SPb.YhTrain) && isequal(SP.YhTest,SPb.YhTest));
    val=struct('X',SP.Xval,'Yh',SP.YhVal,'Ys',SP.YsVal);
    valb=struct('X',SPb.Xval,'Yh',SPb.YhVal,'Ys',SPb.YsVal);
    for name=["Transformer-DualEDL","TemporalR-GAT-DualEDL"]
        model=build_model(name,cfg);
        if name=="Transformer-DualEDL", S=SPb; V=valb; else, S=SP; V=val; end
        model=train_model(model,S.Xtrain,S.YhTrain,S.YsTrain,cfg,V);
        R=evaluate_model(model,S.Xtest,S.YhTest,S.YsTest,cfg); M=R.metrics;
        fprintf('  test | F1 %.4f  AUROC %.4f\n',M.F1,M.AUROC);
        rows(end+1,:)={h,char(name),M.Precision,M.Recall,M.F1,M.AUROC,M.AUPRC,M.Brier}; %#ok<AGROW>
    end
end
T=cell2table(rows,'VariableNames',{'Horizon','Model','Precision','Recall','F1','AUROC','AUPRC','Brier'});
end
