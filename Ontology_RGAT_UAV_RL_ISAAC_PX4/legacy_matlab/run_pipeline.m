function out = run_pipeline(varargin)
%RUN_PIPELINE One command that runs the whole external experiment.
%   Starts the Micro XRCE-DDS agent, Isaac Sim + Pegasus + PX4 SITL and the
%   gateway, checks the link with one expert episode, runs dataset generation,
%   R-GAT training, both PPO runs, the paired evaluation and the plots, then
%   shuts down whatever it started. Processes that were already running are
%   adopted and left running.
%
%   run_pipeline()                      quick mode, headless, full teardown
%   run_pipeline('Mode','full')         the long sweep (hours: PX4 is real time)
%   run_pipeline('UseRunningStack',true)   attach to a stack you started yourself
%   run_pipeline('Headless',true)       no Isaac window (long unattended runs)
%   run_pipeline('KeepStack',true)      leave the simulator up afterwards
%   run_pipeline('SmokeTest',false)     skip the pre-flight episode
%   run_pipeline('SmokeTestOnly',true)  bring the stack up, fly one episode,
%                                       tear it down; no training
%
%   Quick mode flies roughly 520 episodes and full mode roughly 8200 after the
%   moving-deck and battery sweeps. PX4 runs in real time, so budget hours for
%   quick and days for full. If the simulator
%   stops accepting arm commands mid-run, run_pipeline cycles it and continues
%   (cfg.external.resetRecoveries).
%
%   ISAACSIM_PATH must point at the Isaac Sim release directory (the one with
%   python.sh), either in the environment or via the 'IsaacSimPath' option.
%
%   See docs/OPERATIONS.md for the startup order this automates, and the
%   README's "Known limitations" before reading the numbers it produces.

p = inputParser;
p.addParameter('Mode','quick',@(x)any(strcmpi(x,{'quick','full'})));
p.addParameter('Target','sitl',@(x)any(strcmpi(x,{'sitl','hardware'})));
p.addParameter('IsaacSimPath',getenv('ISAACSIM_PATH'),@(x)ischar(x)||isstring(x));
% 'auto' opens the Isaac window whenever there is a display to open it on.
p.addParameter('Headless','auto');
p.addParameter('UseRunningStack',false,@(x)islogical(x)&&isscalar(x));
p.addParameter('KeepStack',false,@(x)islogical(x)&&isscalar(x));
p.addParameter('SmokeTest',true,@(x)islogical(x)&&isscalar(x));
p.addParameter('SmokeTestOnly',false,@(x)islogical(x)&&isscalar(x));
p.parse(varargin{:});
opt = p.Results;

if strcmpi(opt.Target,'hardware')
    error('run_pipeline:hardware', ...
        ['This entry point arms and flies the vehicle unattended and is ' ...
        'SITL only. Use run_hardware_policy.m for a real vehicle.']);
end

setup_external_path();
cfg = defaultExternalConfig(opt.Mode,opt.Target);

% Drop any stack left registered by an interrupted run. Besides leaking a dead
% handle, holding one pins stack.ExternalStack in memory, which stops MATLAB
% reloading the class after it changes on disk.
stack.current([]);

if opt.UseRunningStack
    fprintf('Using the stack that is already running.\n');
    theStack = [];
else
    theStack = stack.ExternalStack(cfg, ...
        'IsaacSimPath',opt.IsaacSimPath,'Headless',opt.Headless);
    if ~opt.KeepStack
        % Guarantees teardown on error and on Ctrl-C, not just on success.
        guard = onCleanup(@()delete(theStack));
    end
    theStack.start();
    % Let sim.resetState cycle this simulator if PX4 stops arming mid-sweep.
    stack.current(theStack);
    restore = onCleanup(@()stack.current([]));
end

if opt.SmokeTest || opt.SmokeTestOnly
    % One expert episode costs about a minute and catches a broken link,
    % a mis-scaled action mapping or a refused arm before hours of training.
    fprintf('\n=== 0) Pre-flight episode ===\n');
    log = sim.runEpisode(struct('type','expert','deterministic',true), ...
        'sparse',[],cfg.seed,cfg,false);
    disp(log.metrics);
    save(fullfile(cfg.paths.results,'external_smoke_test.mat'),'log','cfg');
end

if opt.SmokeTestOnly
    out = struct('smokeTest',log,'cfg',cfg);
    fprintf('\nPre-flight only: the stack came up and one episode flew.\n');
    return;
end

out = pipeline.runAll(cfg);
out.cfg = cfg;

if ~isempty(theStack) && opt.KeepStack
    fprintf('Stack left running by request; stop it with delete(theStack).\n');
    out.stack = theStack;
end
end
