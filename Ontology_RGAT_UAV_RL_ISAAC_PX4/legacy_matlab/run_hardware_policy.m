%% Hardware runtime. Never arms the vehicle; pilot/QGroundControl must arm it.
clear; clc;
setup_external_path();
cfg=defaultExternalConfig('quick','hardware');
p=load(fullfile(cfg.paths.models,'ppo_rgats_pbrs_external.mat'),'proposedAgent');
assert(size(extractdata(p.proposedAgent.actor.W1),2)==cfg.rl.obsDim, ...
    ['Policy input does not match the 20-element moving-pad observation. ' ...
     'Never transfer the old fixed-pad policy to hardware.']);
policy=struct('type','ppo','agent',p.proposedAgent,'deterministic',true);

fprintf(2,['HARDWARE MODE: propellers must have been removed for initial tests.\n' ...
    'RC/QGroundControl takeover and PX4 offboard-loss failsafe must be active.\n']);
b=bridge.PX4Bridge(cfg);
cleanup=onCleanup(@()yieldOffboard(b));
s=b.waitValidState();
s=waitHardwareBattery(b,s,cfg.external.estimatorWarmup);
if ~logical(s.armed)
    error(['Vehicle is not armed. Arm only through the pilot/QGroundControl path, ' ...
        'then restart this script.']);
end
if s.marker_quality<=0
    warning('Marker quality is zero; verify the landing perception publisher.');
end
[x,d]=sim.stateToModel(s,cfg);
env=struct('x',x,'t',0,'step',0,'prevSem',[],'lastDiag',d,'done',false, ...
    'seed',NaN,'bridge',b,'externalState',s,'hasBeenAirborne',~logical(s.landed));
b.enableOffboard();

for k=1:cfg.sim.maxSteps
    cur=sim.getCurrent(env,cfg);
    action=training.policyAction(policy,cur,env,cfg);
    [env,~,~,done,info]=sim.step(env,action,cur,'sparse',[],cfg);
    if mod(k,10)==0
        fprintf(['t=%5.2f z=%6.2f xy=%5.2f marker=%.2f battery=%.1f%% ' ...
            'armed=%d nav=%d\n'], ...
            env.t,env.x(3),norm(env.x(1:2)),env.lastDiag.markerQuality, ...
            100*env.lastDiag.battery.stateOfCharge, ...
            info.armed,info.navState);
    end
    if ~info.armed || done, break; end
end
fprintf('Hardware policy stopped: %s. Pilot/PX4 fallback now owns the vehicle.\n',info.status);
clear cleanup;

function yieldOffboard(client)
try
    if isvalid(client), client.disableOffboard(); end
catch
end
end

function state=waitHardwareBattery(client,state,timeout)
% Never fly an energy-aware hardware policy on the seeded SITL stand-in.
t0=tic;
while toc(t0)<timeout
    if isfield(state,'battery') && isstruct(state.battery) && ...
            isfield(state.battery,'source') && strcmpi(state.battery.source,'px4')
        return;
    end
    pause(0.05); state=client.getState();
end
error(['No live PX4 battery_status reached the gateway. The energy-aware ' ...
    'hardware policy refuses to substitute the SITL battery model.']);
end
