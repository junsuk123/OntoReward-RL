clear; clc; close all;

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(rootDir));

cfg = defaultConfig(rootDir);
cfg.data.mode = "NCLT";
cfg.data.session = "2013-01-10";
cfg.visual.live = true;

runExperiment(cfg);

