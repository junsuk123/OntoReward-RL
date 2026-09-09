clear; clc; close all;

rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(rootDir));

cfg = defaultConfig(rootDir);
cfg.data.mode = "SYNTHETIC";
cfg.data.maxSamples = 900;
cfg.learning.epochs = 8;
cfg.learning.batchSize = 12;
cfg.visual.live = true;

runExperiment(cfg);
