function R = evaluate_model(model,X,Yh,Ys,cfg)
%EVALUATE_MODEL Inference, metrics, predictions.
[p,u]=predict_probabilities(model,X,cfg);
thr=cfg.threshold;
if isfield(model,"threshold") && ~isempty(model.threshold), thr=model.threshold; end
M=compute_metrics(Yh,p,thr);
M.Threshold=thr;
pred=p>=thr; correct=(pred(:)==logical(Yh(:)));
if any(isfinite(u))
    M.MeanUncertainty=mean(double(u(isfinite(u))));
    M.MeanUncertaintyCorrect=mean(double(u(correct' & isfinite(u))),'omitnan');
    M.MeanUncertaintyIncorrect=mean(double(u((~correct)' & isfinite(u))),'omitnan');
else
    M.MeanUncertainty=NaN; M.MeanUncertaintyCorrect=NaN; M.MeanUncertaintyIncorrect=NaN;
end
R=struct("name",model.name,"pFault",p,"uncertainty",u,"metrics",M, ...
    "Yhard",Yh,"Ysoft",Ys,"threshold",thr);
end
