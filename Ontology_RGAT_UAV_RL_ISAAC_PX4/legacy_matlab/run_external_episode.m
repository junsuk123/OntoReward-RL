%% Evaluate the trained proposed policy against Isaac/PX4.
clear; clc;
setup_external_path();
cfg=defaultExternalConfig('quick','sitl');
p=load(fullfile(cfg.paths.models,'ppo_rgats_pbrs_external.mat'),'proposedAgent');
r=load(fullfile(cfg.paths.models,'rgat_model_external.mat'),'rgatModel');
assert(size(extractdata(p.proposedAgent.actor.W1),2)==cfg.rl.obsDim, ...
    'Policy input does not match the 20-element moving-pad observation. Retrain externally.');
assert(size(extractdata(r.rgatModel.W1),2)==cfg.ontology.inDim, ...
    'R-GAT input does not match the 13-node moving-pad ontology. Retrain externally.');
policy=struct('type','ppo','agent',p.proposedAgent,'deterministic',true);
log=sim.runEpisode(policy,'proposed',r.rgatModel,cfg.seed,cfg,true);
disp(log.metrics);
save(fullfile(cfg.paths.results,'external_proposed_episode.mat'),'log','cfg');
