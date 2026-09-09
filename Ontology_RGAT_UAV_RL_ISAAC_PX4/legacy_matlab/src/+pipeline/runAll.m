function out = runAll(cfg)
%RUNALL Dataset, R-GAT, both PPO runs, evaluation and plots against Isaac/PX4.
%   Assumes the external stack is already up; stack.ExternalStack starts it.
%   Every stage writes its artefact before the next one begins, so a failure
%   late in a long sweep does not throw away the hours before it.

rng(cfg.seed,'twister');
t0 = tic;

fprintf('\n=== 1) External stack connectivity ===\n');
probe = bridge.PX4Bridge(cfg);
state = probe.waitValidState();
fprintf('PX4 state valid: z=%.2f m, armed=%d, source=%s\n', ...
    state.position(3),state.armed,state.source);
delete(probe);

fprintf('\n=== 2) Isaac/PX4 R-GAT dataset generation ===\n');
D = training.generateRGATDataset(cfg);
save(fullfile(cfg.paths.data,'rgat_dataset_external.mat'),'D','-v7.3');

fprintf('\n=== 3) R-GAT potential training ===\n');
fprintf(['The simulator remains online, but the UAV stays landed during this ' ...
    'offline neural-network training stage.\n']);
[rgatModel,rgatHistory] = training.trainRGAT(D,cfg);
save(fullfile(cfg.paths.models,'rgat_model_external.mat'),'rgatModel','rgatHistory','cfg');

fprintf('\n=== 4) PPO manual baseline on PX4 ===\n');
[baselineAgent,baselineHistory] = training.trainPPO('manual',[],cfg);
save(fullfile(cfg.paths.models,'ppo_manual_external.mat'), ...
    'baselineAgent','baselineHistory','cfg');

fprintf('\n=== 5) PPO ontology-RGAT PBRS on PX4 ===\n');
[proposedAgent,proposedHistory] = training.trainPPO('proposed',rgatModel,cfg);
save(fullfile(cfg.paths.models,'ppo_rgats_pbrs_external.mat'), ...
    'proposedAgent','proposedHistory','cfg','rgatModel');

fprintf('\n=== 6) Paired external evaluation ===\n');
results = evaluation.comparePolicies(baselineAgent,proposedAgent,rgatModel,cfg);
results.windSweep = evaluation.windSweep(baselineAgent,proposedAgent,rgatModel,cfg);
save(fullfile(cfg.paths.results,'comparison_results_external.mat'),'results','cfg');
writetable(results.summary,fullfile(cfg.paths.results,'summary_metrics_external.csv'));
writetable(results.perEpisode,fullfile(cfg.paths.results,'episode_metrics_external.csv'));

fprintf('\n=== 7) Publication plots ===\n');
evaluation.makePlots(results,baselineHistory,proposedHistory,cfg);

out = struct('rgatModel',rgatModel,'rgatHistory',rgatHistory, ...
    'baselineAgent',baselineAgent,'baselineHistory',baselineHistory, ...
    'proposedAgent',proposedAgent,'proposedHistory',proposedHistory, ...
    'results',results,'elapsedSeconds',toc(t0));
fprintf('\nExternal pipeline complete in %.1f min: %s\n', ...
    out.elapsedSeconds/60,cfg.paths.results);
end
