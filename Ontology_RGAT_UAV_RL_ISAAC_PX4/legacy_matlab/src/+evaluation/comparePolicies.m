function R = comparePolicies(baselineAgent,proposedAgent,rgatModel,cfg)
%COMPAREPOLICIES Paired common-random-number comparison on the external stack.
%   In addition to the fixed-pad metrics, report horizontal touchdown speed
%   relative to the moving deck, the deck speed, and the measured episode
%   energy. Proposed-minus-manual confidence intervals remain paired by seed.
B=evaluation.evaluatePolicy(baselineAgent,'Manual',rgatModel,cfg);
P=evaluation.evaluatePolicy(proposedAgent,'Ontology-RGAT',rgatModel,cfg);
N=cfg.eval.episodes;

metricNames={'Success','Unsafe','Timeout','Depleted','TouchdownXY','TouchdownVz', ...
    'TouchdownRelSpeedXY','MaxTiltDeg','EnergyJ','MaxAeroForce', ...
    'PadSpeedMean','BatteryReserveFinal'};
metricFields={'success','unsafe','timeout','depleted','touchdownXY','touchdownVz', ...
    'touchdownRelSpeedXY','maxTilt','energyJ','maxAeroForce', ...
    'padSpeedMean','batteryReserveFinal'};
Policy=strings(2*N,1); Seed=zeros(2*N,1);
values=zeros(2*N,numel(metricFields));
for j=1:2
    if j==1, X=B; else, X=P; end
    for i=1:N
        row=(j-1)*N+i; m=X.metrics{i};
        Policy(row)=X.label; Seed(row)=m.seed;
        for k=1:numel(metricFields)
            value=m.(metricFields{k});
            if strcmp(metricFields{k},'maxTilt'), value=rad2deg(value); end
            values(row,k)=value;
        end
    end
end

perEpisode=table(Policy,Seed);
for k=1:numel(metricNames)
    perEpisode.(metricNames{k})=values(:,k);
end

labels=["Manual";"Ontology-RGAT"];
summary=table(labels,'VariableNames',{'Policy'});
rateMetrics={'Success','Unsafe','Timeout','Depleted'};
for k=1:numel(metricNames)
    column=zeros(2,1);
    for j=1:2
        column(j)=mean(values(Policy==labels(j),k));
    end
    if ismember(metricNames{k},rateMetrics)
        outputName=[metricNames{k} 'Rate'];
    else
        outputName=['Mean' metricNames{k}];
    end
    summary.(outputName)=column;
end
summary.Episodes=[sum(Policy==labels(1));sum(Policy==labels(2))];

% Paired common-random-number differences: proposed minus manual.
Metric=strings(numel(metricNames),1); MeanDifference=zeros(numel(metricNames),1);
CI95Low=zeros(numel(metricNames),1); CI95High=zeros(numel(metricNames),1);
manual=values(Policy==labels(1),:); proposed=values(Policy==labels(2),:);
for k=1:numel(metricNames)
    difference=proposed(:,k)-manual(:,k);
    mu=mean(difference); se=std(difference)/sqrt(max(1,numel(difference)));
    Metric(k)=metricNames{k}; MeanDifference(k)=mu;
    CI95Low(k)=mu-1.96*se; CI95High(k)=mu+1.96*se;
end
paired=table(Metric,MeanDifference,CI95Low,CI95High);

R=struct('baseline',B,'proposed',P,'perEpisode',perEpisode, ...
    'summary',summary,'paired',paired,'proposedModel',rgatModel);
writetable(paired,fullfile(cfg.paths.results,'paired_difference_ci.csv'));
end
