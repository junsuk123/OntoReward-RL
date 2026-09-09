%% Integrated ontology/R-GAT/PPO pipeline against an already-running stack.
%  Use run_pipeline for a one-command run that also starts and stops Isaac,
%  PX4 and the gateway.
clear; clc; close all;
setup_external_path();
cfg=defaultExternalConfig('full','sitl'); % use 'full' only after short runs pass
out=pipeline.runAll(cfg); %#ok<NASGU>
