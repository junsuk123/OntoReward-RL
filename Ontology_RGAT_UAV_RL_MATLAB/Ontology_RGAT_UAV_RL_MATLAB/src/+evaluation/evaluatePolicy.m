function out = evaluatePolicy(agent,label,rgatModel,cfg)
%EVALUATEPOLICY Deterministic paired-seed Monte Carlo using common sparse task reward.
N=cfg.eval.episodes; m=cell(N,1); logs=cell(N,1);
policy=struct('type','ppo','agent',agent,'deterministic',true);
for i=1:N
    seed=cfg.eval.seed0+i-1;
    L=sim.runEpisode(policy,'sparse',rgatModel,seed,cfg,false);
    m{i}=L.metrics;
    if i==1, logs{i}=L; end
    fprintf('Eval %-10s %3d/%3d | success=%d | xy=%.3f | vz=%.3f\n',label,i,N,L.metrics.success,L.metrics.touchdownXY,L.metrics.touchdownVz);
end
out.label=label; out.metrics=m; out.representative=logs{1};
end
