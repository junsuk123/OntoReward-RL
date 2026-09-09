function T=padSweep(baselineAgent,proposedAgent,rgatModel,cfg)
%PADSWEEP How fast a deck each policy can still land on.
%   The moving-pad counterpart of evaluation.windSweep: the seed picks the same
%   point in the deck-motion distribution at every scale, so the comparison is
%   paired across speeds as well as across policies.
scales=cfg.eval.padScales(:); ns=numel(scales); N=cfg.eval.sweepEpisodes;
Policy=strings(2*ns,1); PadScale=zeros(2*ns,1); MeanPadSpeed=zeros(2*ns,1);
SuccessRate=zeros(2*ns,1); UnsafeRate=zeros(2*ns,1); TimeoutRate=zeros(2*ns,1);
DepletedRate=zeros(2*ns,1); MeanXY=zeros(2*ns,1); MeanRelSpeedXY=zeros(2*ns,1);
row=0;
for si=1:ns
    c=cfg; s=scales(si); c.external.padScale=s;
    agents={baselineAgent,proposedAgent}; labels={"Manual","Ontology-RGAT"};
    for j=1:2
        succ=zeros(N,1); unsafe=zeros(N,1); tout=zeros(N,1); dep=zeros(N,1);
        xy=zeros(N,1); rel=zeros(N,1); ps=zeros(N,1);
        policy=struct('type','ppo','agent',agents{j},'deterministic',true);
        for i=1:N
            seed=cfg.eval.seed0+20000*si+i;
            L=sim.runEpisode(policy,'sparse',rgatModel,seed,c,false);
            succ(i)=L.metrics.success; unsafe(i)=L.metrics.unsafe;
            tout(i)=L.metrics.timeout; dep(i)=L.metrics.depleted;
            xy(i)=L.metrics.touchdownXY; rel(i)=L.metrics.touchdownRelSpeedXY;
            ps(i)=L.metrics.padSpeedMean;
        end
        row=row+1; Policy(row)=labels{j}; PadScale(row)=s;
        MeanPadSpeed(row)=mean(ps);
        SuccessRate(row)=mean(succ); UnsafeRate(row)=mean(unsafe);
        TimeoutRate(row)=mean(tout); DepletedRate(row)=mean(dep);
        MeanXY(row)=mean(xy); MeanRelSpeedXY(row)=mean(rel);
        fprintf(['Pad x%.2f (%.2f m/s) | %-13s | success %.1f%% | unsafe %.1f%% | ' ...
            'timeout %.1f%% | depleted %.1f%% | rel speed %.2f m/s\n'], ...
            s,MeanPadSpeed(row),labels{j},100*SuccessRate(row),100*UnsafeRate(row), ...
            100*TimeoutRate(row),100*DepletedRate(row),MeanRelSpeedXY(row));
    end
end
T=table(Policy,PadScale,MeanPadSpeed,SuccessRate,UnsafeRate,TimeoutRate, ...
    DepletedRate,MeanXY,MeanRelSpeedXY);
writetable(T,fullfile(cfg.paths.results,'pad_speed_sweep.csv'));
end
