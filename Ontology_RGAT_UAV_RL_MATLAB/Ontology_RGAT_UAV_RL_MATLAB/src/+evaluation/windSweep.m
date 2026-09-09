function T = windSweep(baselineAgent,proposedAgent,rgatModel,cfg)
%WINDSWEEP Evaluate both policies across unseen disturbance intensity scaling.
scales=cfg.eval.windScales(:); ns=numel(scales); N=cfg.eval.sweepEpisodes;
Policy=strings(2*ns,1); WindScale=zeros(2*ns,1); SuccessRate=zeros(2*ns,1);
UnsafeRate=zeros(2*ns,1); TimeoutRate=zeros(2*ns,1); MeanXY=zeros(2*ns,1);
MeanMaxTiltDeg=zeros(2*ns,1);
row=0;
for si=1:ns
    c=cfg; s=scales(si);
    c.wind.meanRef=cfg.wind.meanRef*s;
    c.wind.modes.amp=cfg.wind.modes.amp*s;
    for g=1:numel(c.wind.gusts), c.wind.gusts(g).vector=cfg.wind.gusts(g).vector*s; end
    c.wind.vortex.strength=cfg.wind.vortex.strength*s;
    agents={baselineAgent,proposedAgent}; labels={"Manual","Ontology-RGAT"};
    for j=1:2
        succ=zeros(N,1); unsafe=zeros(N,1); tout=zeros(N,1); xy=zeros(N,1); mt=zeros(N,1);
        policy=struct('type','ppo','agent',agents{j},'deterministic',true);
        for i=1:N
            seed=cfg.eval.seed0+10000*si+i;
            L=sim.runEpisode(policy,'sparse',rgatModel,seed,c,false);
            succ(i)=L.metrics.success; unsafe(i)=L.metrics.unsafe; tout(i)=L.metrics.timeout;
            xy(i)=L.metrics.touchdownXY; mt(i)=rad2deg(L.metrics.maxTilt);
        end
        row=row+1; Policy(row)=labels{j}; WindScale(row)=s;
        SuccessRate(row)=mean(succ); UnsafeRate(row)=mean(unsafe); TimeoutRate(row)=mean(tout);
        MeanXY(row)=mean(xy); MeanMaxTiltDeg(row)=mean(mt);
        fprintf('Wind x%.2f | %-13s | success %.1f%% | unsafe %.1f%% | timeout %.1f%%\n', ...
            s,labels{j},100*SuccessRate(row),100*UnsafeRate(row),100*TimeoutRate(row));
    end
end
T=table(Policy,WindScale,SuccessRate,UnsafeRate,TimeoutRate,MeanXY,MeanMaxTiltDeg);
writetable(T,fullfile(cfg.paths.results,'wind_generalization_sweep.csv'));
end
