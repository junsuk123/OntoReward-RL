function cfg = defaultExternalConfig(mode,target)
%DEFAULTEXTERNALCONFIG Original learning configuration + external flight stack.
if nargin<1, mode='quick'; end
if nargin<2, target='sitl'; end
root=setup_external_path();
cfg=defaultConfig(mode);
cfg.external.enabled=true;
cfg.external.target=lower(target);
cfg.external.protocolVersion=1;
cfg.external.gatewayHost='127.0.0.1';
cfg.external.gatewayPort=14650;
cfg.external.localHost='127.0.0.1';
cfg.external.localPort=14651;
cfg.external.timeout=2.0;
cfg.external.estimatorWarmup=5.0;
cfg.external.actionTimeout=0.25;
cfg.external.autoArm=strcmpi(target,'sitl');
cfg.external.prestreamCount=24;
cfg.external.controlHz=50.0;
cfg.external.resetSettle=0.15;
% PX4 flies the seeded entry pose before the policy takes over; hand over only
% once it is actually holding that point.
cfg.external.entryTolerance=0.45;      % m
cfg.external.entrySpeedTolerance=0.35; % m/s
cfg.external.entrySettle=0.5;          % s held inside tolerance
cfg.external.entryTimeout=90.0;        % s, must outlast PX4's post-boot arm refusal
cfg.external.armRetry=2.0;             % s between arm attempts during the climb
% PX4 SITL stops accepting arm commands after hours of lockstep; let a
% pipeline that owns the simulator cycle it instead of losing the run.
cfg.external.resetRecoveries=2;
cfg.external.windScale=1.0;
cfg.external.useEstimatedState=true;
cfg.sim.dt=0.02;
cfg.sim.maxSteps=round(cfg.sim.maxTime/cfg.sim.dt);
% PX4 already fuses sensor noise; do not add a second synthetic noise layer.
cfg.sensor.posStd=0; cfg.sensor.velStd=0;
cfg.sensor.angleStd=0; cfg.sensor.rateStd=0;
cfg.paths.root=root;
cfg.paths.results=fullfile(root,'results');
cfg.paths.models=fullfile(cfg.paths.results,'models');
cfg.paths.figures=fullfile(cfg.paths.results,'figures');
cfg.paths.data=fullfile(cfg.paths.results,'data');
cfg.paths.live=fullfile(cfg.paths.results,'live');
for d={cfg.paths.results,cfg.paths.models,cfg.paths.figures,cfg.paths.data,cfg.paths.live}
    if ~isfolder(d{1}), mkdir(d{1}); end
end
end
