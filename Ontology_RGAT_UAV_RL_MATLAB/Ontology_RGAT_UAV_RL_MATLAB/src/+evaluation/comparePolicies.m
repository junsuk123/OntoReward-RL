function R = comparePolicies(baselineAgent,proposedAgent,rgatModel,cfg)
B=evaluation.evaluatePolicy(baselineAgent,'Manual',rgatModel,cfg);
P=evaluation.evaluatePolicy(proposedAgent,'Ontology-RGAT',rgatModel,cfg);
N=cfg.eval.episodes;
Policy=strings(2*N,1); Seed=zeros(2*N,1); Success=zeros(2*N,1); Unsafe=zeros(2*N,1);
Timeout=zeros(2*N,1); TouchdownXY=zeros(2*N,1); TouchdownVz=zeros(2*N,1);
MaxTiltDeg=zeros(2*N,1); EnergyJ=zeros(2*N,1); MaxAeroForce=zeros(2*N,1);
for j=1:2
    if j==1, X=B; else, X=P; end
    for i=1:N
        r=(j-1)*N+i; m=X.metrics{i}; Policy(r)=X.label; Seed(r)=m.seed;
        Success(r)=m.success; Unsafe(r)=m.unsafe; Timeout(r)=m.timeout;
        TouchdownXY(r)=m.touchdownXY; TouchdownVz(r)=m.touchdownVz;
        MaxTiltDeg(r)=rad2deg(m.maxTilt);
        EnergyJ(r)=m.energyJ; MaxAeroForce(r)=m.maxAeroForce;
    end
end
perEpisode=table(Policy,Seed,Success,Unsafe,Timeout,TouchdownXY,TouchdownVz,MaxTiltDeg,EnergyJ,MaxAeroForce);
labels=["Manual";"Ontology-RGAT"];
vt=repmat({'double'},1,9); vt=[{'string'},vt];
summary=table('Size',[2,10],'VariableTypes',vt, ...
    'VariableNames',{'Policy','SuccessRate','UnsafeRate','TimeoutRate','MeanTouchdownXY', ...
                     'MeanTouchdownVz','MeanMaxTiltDeg','MeanEnergyJ','MeanMaxAeroForce','Episodes'});
for j=1:2
    ix=Policy==labels(j); summary.Policy(j)=labels(j);
    summary.SuccessRate(j)=mean(Success(ix)); summary.UnsafeRate(j)=mean(Unsafe(ix));
    summary.TimeoutRate(j)=mean(Timeout(ix));
    summary.MeanTouchdownXY(j)=mean(TouchdownXY(ix)); summary.MeanTouchdownVz(j)=mean(TouchdownVz(ix));
    summary.MeanMaxTiltDeg(j)=mean(MaxTiltDeg(ix)); summary.MeanEnergyJ(j)=mean(EnergyJ(ix));
    summary.MeanMaxAeroForce(j)=mean(MaxAeroForce(ix)); summary.Episodes(j)=sum(ix);
end
% Paired common-random-number differences: Proposed - Manual.
mb=perEpisode(Policy=="Manual",:); mp=perEpisode(Policy=="Ontology-RGAT",:);
metricNames={'Success','Unsafe','Timeout','TouchdownXY','TouchdownVz','MaxTiltDeg','EnergyJ','MaxAeroForce'};
Metric=strings(numel(metricNames),1); MeanDifference=zeros(numel(metricNames),1);
CI95Low=zeros(numel(metricNames),1); CI95High=zeros(numel(metricNames),1);
for k=1:numel(metricNames)
    d=mp.(metricNames{k})-mb.(metricNames{k}); mu=mean(d); se=std(d)/sqrt(max(1,numel(d)));
    Metric(k)=metricNames{k}; MeanDifference(k)=mu; CI95Low(k)=mu-1.96*se; CI95High(k)=mu+1.96*se;
end
paired=table(Metric,MeanDifference,CI95Low,CI95High);
R=struct('baseline',B,'proposed',P,'perEpisode',perEpisode,'summary',summary,'paired',paired,'proposedModel',rgatModel);
writetable(paired,fullfile(cfg.paths.results,'paired_difference_ci.csv'));
end
