%% Ontology + R-GAT reward shaping for 6-DOF UAV landing
clear; clc; close all;
root = setup_path(); %#ok<NASGU>
cfg = defaultConfig('full');  % change to 'full' for paper-scale runs
rng(cfg.seed,'twister');

fprintf('\n=== 1) Physics validation ===\n');
validationReport = validatePhysics(cfg);
save(fullfile(cfg.paths.results,'validation_report.mat'),'validationReport');

fprintf('\n=== 2) R-GAT dataset generation ===\n');
datasetFile = fullfile(cfg.paths.data,'rgat_dataset.mat');
D = training.generateRGATDataset(cfg);
save(datasetFile,'D','-v7.3');

fprintf('\n=== 3) R-GAT potential model training ===\n');
[rgatModel, rgatHistory] = training.trainRGAT(D,cfg);
save(fullfile(cfg.paths.models,'rgat_model.mat'),'rgatModel','rgatHistory','cfg');

fprintf('\n=== 4) PPO baseline: hand-designed dense reward ===\n');
[baselineAgent, baselineHistory] = training.trainPPO('manual',[],cfg);
save(fullfile(cfg.paths.models,'ppo_manual.mat'),'baselineAgent','baselineHistory','cfg');

fprintf('\n=== 5) PPO proposed: ontology R-GAT PBRS ===\n');
[proposedAgent, proposedHistory] = training.trainPPO('proposed',rgatModel,cfg);
save(fullfile(cfg.paths.models,'ppo_rgats_pbrs.mat'),'proposedAgent','proposedHistory','cfg','rgatModel');

fprintf('\n=== 6) Paired Monte-Carlo evaluation ===\n');
results = evaluation.comparePolicies(baselineAgent,proposedAgent,rgatModel,cfg);
fprintf('\n=== 6b) Wind-intensity generalization sweep ===\n');
results.windSweep = evaluation.windSweep(baselineAgent,proposedAgent,rgatModel,cfg);
save(fullfile(cfg.paths.results,'comparison_results.mat'),'results','cfg');
writetable(results.summary,fullfile(cfg.paths.results,'summary_metrics.csv'));
writetable(results.perEpisode,fullfile(cfg.paths.results,'episode_metrics.csv'));

fprintf('\n=== 7) Publication plots ===\n');
evaluation.makePlots(results,baselineHistory,proposedHistory,cfg);

fprintf('\nFinished. Results: %s\n',cfg.paths.results);
