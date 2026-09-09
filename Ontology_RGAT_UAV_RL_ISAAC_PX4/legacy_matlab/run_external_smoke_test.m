%% One guarded flight-stack-in-the-loop episode with the expert policy.
clear; clc;
setup_external_path();
cfg=defaultExternalConfig('quick','sitl');
assert(strcmpi(cfg.external.target,'sitl'), ...
    'Smoke test refuses non-SITL targets.');
policy=struct('type','expert','deterministic',true);
log=sim.runEpisode(policy,'sparse',[],cfg.seed,cfg,false);
disp(log.metrics);
save(fullfile(cfg.paths.results,'external_smoke_test.mat'),'log','cfg');

