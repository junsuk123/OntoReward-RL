%% Real-time monitored evaluation demo
clear; clc; close all;
root=setup_path();
tmp=defaultConfig('quick');
modelFile=fullfile(tmp.paths.models,'ppo_rgats_pbrs.mat');
if ~exist(modelFile,'file')
    error('Train first with run_all.m (missing %s).', modelFile);
end
S=load(modelFile,'proposedAgent','rgatModel','cfg');
cfg=S.cfg; cfg.sim.realtime=true;
policy=struct('type','ppo','agent',S.proposedAgent,'deterministic',true);
log=sim.runEpisode(policy,'proposed',S.rgatModel,cfg.eval.seed0,cfg,true);
disp(log.metrics);
