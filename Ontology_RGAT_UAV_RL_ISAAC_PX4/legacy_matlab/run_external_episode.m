%% Evaluate the trained proposed policy against Isaac/PX4.
clear; clc;
root=setup_external_path();
cfg=defaultExternalConfig('quick','sitl');
original=fullfile(fileparts(root),'Ontology_RGAT_UAV_RL_MATLAB', ...
    'Ontology_RGAT_UAV_RL_MATLAB','results','models');
p=load(fullfile(original,'ppo_rgats_pbrs.mat'),'proposedAgent');
r=load(fullfile(original,'rgat_model.mat'),'rgatModel');
policy=struct('type','ppo','agent',p.proposedAgent,'deterministic',true);
log=sim.runEpisode(policy,'proposed',r.rgatModel,cfg.seed,cfg,true);
disp(log.metrics);
save(fullfile(cfg.paths.results,'external_proposed_episode.mat'),'log','cfg');

