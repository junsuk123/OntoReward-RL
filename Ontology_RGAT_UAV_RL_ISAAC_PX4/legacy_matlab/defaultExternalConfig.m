function cfg = defaultExternalConfig(mode,target)
%DEFAULTEXTERNALCONFIG Original learning configuration + external flight stack.
%   Extends the read-only original configuration with everything the external
%   backend adds: a landing pad that rides a ground vehicle, an onboard energy
%   budget, and the two ontology nodes those two facts introduce.
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
% once it is actually holding that point. With a moving deck the entry pose is
% an offset in the pad frame and the gateway re-aims it at the live deck, so
% these tolerances are on the pad-relative state, not on a world point.
cfg.external.entryFrame='pad';
cfg.external.entryTolerance=0.45;      % m
cfg.external.entrySpeedTolerance=0.35; % m/s, pad-relative
cfg.external.entrySettle=0.5;          % s held inside tolerance
cfg.external.entryTimeout=90.0;        % s, must outlast PX4's post-boot arm refusal
cfg.external.armRetry=2.0;             % s between arm attempts during the climb
% PX4 SITL stops accepting arm commands after hours of lockstep; let a
% pipeline that owns the simulator cycle it instead of losing the run.
cfg.external.resetRecoveries=2;
cfg.external.windScale=1.0;
% Multiplies the deck speed Isaac draws for the episode; evaluation.padSweep
% varies it the way evaluation.windSweep varies the wind.
cfg.external.padScale=1.0;
cfg.external.useEstimatedState=true;
cfg.sim.dt=0.02;
cfg.sim.maxSteps=round(cfg.sim.maxTime/cfg.sim.dt);
% PX4 already fuses sensor noise; do not add a second synthetic noise layer.
cfg.sensor.posStd=0; cfg.sensor.velStd=0;
cfg.sensor.angleStd=0; cfg.sensor.rateStd=0;

%% Moving landing pad
% The target is a deck on a ground vehicle, so every position and velocity the
% policy sees is pad-relative (see docs/ARCHITECTURE.md, "Coordinates"). The
% deck's motion profile itself lives in config/system.yaml, which Isaac owns;
% these are only the terms the learning side needs.
cfg.pad.enabled=true;
% Matches landing.success_rel_speed_xy_m_s in config/system.yaml. Touching down
% on a moving deck with unmatched horizontal velocity tips the airframe over,
% so closing speed joins the landing criteria instead of being ignored.
cfg.criteria.relSpeedXY=0.45;          % m/s
% Normalization for the PadMotion ontology node and the observation, in the
% same style as the other semantic scales: the speed at which chasing the deck
% dominates the landing problem.
cfg.semantic.padSpeedScale=1.6;        % m/s

%% Onboard energy
% The gateway owns the battery model and reports its parameters in every state
% reply, so nothing here duplicates config/system.yaml; these are the ontology
% and reward scalings only.
cfg.battery.enabled=true;
% Descent rate the margin calculation assumes when pricing "can I still land
% from here", deliberately conservative relative to the expert's profile.
cfg.battery.planDescentRate=0.55;      % m/s
cfg.semantic.energyScale=1.0;          % margin, in units of the episode horizon

%% Ontology graph: two nodes the external backend introduces
cfg.ontology.nodeNames={'PositionError','VerticalSpeed','TiltAngle','AngularRate', ...
    'WindRisk','MarkerQuality','VisualStability','Alignment','AttitudeStability', ...
    'TouchdownSafety','PadMotion','BatteryReserve','SafeLanding'};
cfg.ontology.nNodes=numel(cfg.ontology.nodeNames);
cfg.ontology.nRelations=numel(cfg.ontology.relationNames);
cfg.ontology.inDim=4+cfg.ontology.nNodes;
% pad-relative pose/velocity, attitude, rates, the three original semantic
% channels, then deck velocity feed-forward, deck motion, reserve and margin.
cfg.rl.obsDim=20;

%% Reward terms for the two new factors
% Both arms are judged by the same criteria; only the manual dense baseline
% needs hand-designed weights for them, which is the point of the comparison.
cfg.reward.manual.wPadTrack=0.55;   % penalise pad-relative closing speed
cfg.reward.manual.wEnergy=0.40;     % penalise burning the reserve
cfg.reward.manual.batteryDepleted=-20;
cfg.reward.sparse.batteryDepleted=-10;

%% GPU
% Measured, not assumed. The R-GAT layer is kernel-launch bound rather than
% FLOP bound: at cfg.rgat.batchSize=32 the GPU is about 2x slower than the
% vectorized CPU path on an RTX 4060. Batch 256 is effectively a tie on repeat
% runs, while 1024 is clearly faster on the GPU, so 'auto' uses that conservative
% crossover rather than moving work for a marginal result.
% matlab/tools/benchmark_rgat.m measures the crossover on this machine.
cfg.gpu.mode='auto';                % 'auto' | 'on' | 'off'
cfg.gpu.minBatchForGPU=1024;
cfg.gpu.precision='single';         % GPU only; the CPU path stays double

%% Evaluation sweeps for the two new factors
cfg.eval.padScales=[0.0, 0.5, 1.0, 1.5, 2.0];   % 0.0 is the static control condition
cfg.eval.batteryBinsS=[0 10 20 30 50];          % starting reserve, hover seconds

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
