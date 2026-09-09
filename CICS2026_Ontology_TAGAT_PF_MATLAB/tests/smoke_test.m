function smoke_test
%SMOKE_TEST Fast non-publication pipeline check.

rootDir=fileparts(fileparts(mfilename("fullpath")));
addpath(genpath(rootDir));
cfg=defaultConfig(rootDir);
cfg.data.mode="SYNTHETIC";
cfg.data.maxSamples=240;
cfg.learning.epochs=1;
cfg.learning.batchSize=8;
cfg.visual.live=false;
cfg.pf.initialN=200;
cfg.pf.fixedN=250;
cfg.pf.minN=120;
cfg.pf.maxN=450;
cfg.pf.baseN=200;

summary=runExperiment(cfg);
assert(height(summary.metrics)>=3);
disp("Smoke test completed.");
end
