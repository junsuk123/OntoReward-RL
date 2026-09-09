%% Fast structural smoke test (no neural training)
clear; clc; close all; setup_path();
cfg = defaultConfig('quick');
validatePhysics(cfg);
policy = struct('type','expert_noisy','noiseStd',0.05,'deterministic',true);
log = sim.runEpisode(policy,'manual',[],123,cfg,false);
fprintf('Smoke episode: success=%d, steps=%d, touchdownXY=%.3f m\n', ...
    log.metrics.success,log.metrics.steps,log.metrics.touchdownXY);
