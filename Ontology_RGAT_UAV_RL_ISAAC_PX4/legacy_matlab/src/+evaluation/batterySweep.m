function T=batterySweep(baselineAgent,proposedAgent,rgatModel,cfg)
%BATTERYSWEEP Success against how much reserve the episode started with.
%   The reserve is seeded per episode, not commanded, so this bins finished
%   episodes by the starting reserve instead of sweeping a scale: that is the
%   honest way to read a factor the initial condition owns.
N=cfg.eval.sweepEpisodes*numel(cfg.eval.padScales);
edges=cfg.eval.batteryBinsS(:)';
agents={baselineAgent,proposedAgent}; labels={"Manual","Ontology-RGAT"};
nb=numel(edges)-1;
Policy=strings(2*nb,1); ReserveLowS=zeros(2*nb,1); ReserveHighS=zeros(2*nb,1);
Episodes=zeros(2*nb,1); SuccessRate=zeros(2*nb,1); DepletedRate=zeros(2*nb,1);
UnsafeRate=zeros(2*nb,1); MeanEnergyJ=zeros(2*nb,1);
for j=1:2
    policy=struct('type','ppo','agent',agents{j},'deterministic',true);
    startS=zeros(N,1); succ=zeros(N,1); dep=zeros(N,1); uns=zeros(N,1); eJ=zeros(N,1);
    for i=1:N
        seed=cfg.eval.seed0+30000+i;
        L=sim.runEpisode(policy,'sparse',rgatModel,seed,cfg,false);
        % Starting reserve, reconstructed from the pack's own currency.
        startS(i)=L.hoverSecondsLeft(1);
        succ(i)=L.metrics.success; dep(i)=L.metrics.depleted;
        uns(i)=L.metrics.unsafe; eJ(i)=L.metrics.energyJ;
    end
    for b=1:nb
        ix=startS>=edges(b) & startS<edges(b+1);
        row=(j-1)*nb+b;
        Policy(row)=labels{j}; ReserveLowS(row)=edges(b); ReserveHighS(row)=edges(b+1);
        Episodes(row)=sum(ix);
        if any(ix)
            SuccessRate(row)=mean(succ(ix)); DepletedRate(row)=mean(dep(ix));
            UnsafeRate(row)=mean(uns(ix)); MeanEnergyJ(row)=mean(eJ(ix));
        else
            SuccessRate(row)=NaN; DepletedRate(row)=NaN;
            UnsafeRate(row)=NaN; MeanEnergyJ(row)=NaN;
        end
        fprintf('Reserve %4.1f-%4.1f s | %-13s | n=%3d | success %5.1f%% | depleted %5.1f%%\n', ...
            edges(b),edges(b+1),labels{j},Episodes(row), ...
            100*SuccessRate(row),100*DepletedRate(row));
    end
end
T=table(Policy,ReserveLowS,ReserveHighS,Episodes,SuccessRate,UnsafeRate, ...
    DepletedRate,MeanEnergyJ);
writetable(T,fullfile(cfg.paths.results,'battery_reserve_bins.csv'));
end
