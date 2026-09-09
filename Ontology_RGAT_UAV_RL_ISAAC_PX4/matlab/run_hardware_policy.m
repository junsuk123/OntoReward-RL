%% Hardware runtime. Never arms the vehicle; pilot/QGroundControl must arm it.
clear; clc;
root=setup_external_path();
cfg=defaultExternalConfig('quick','hardware');
original=fullfile(fileparts(root),'Ontology_RGAT_UAV_RL_MATLAB', ...
    'Ontology_RGAT_UAV_RL_MATLAB','results','models');
p=load(fullfile(original,'ppo_rgats_pbrs.mat'),'proposedAgent');
policy=struct('type','ppo','agent',p.proposedAgent,'deterministic',true);

fprintf(2,['HARDWARE MODE: propellers must have been removed for initial tests.\n' ...
    'RC/QGroundControl takeover and PX4 offboard-loss failsafe must be active.\n']);
b=bridge.PX4Bridge(cfg);
cleanup=onCleanup(@()yieldOffboard(b));
s=b.waitValidState();
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
        fprintf('t=%5.2f z=%6.2f xy=%5.2f marker=%.2f armed=%d nav=%d\n', ...
            env.t,env.x(3),norm(env.x(1:2)),env.lastDiag.markerQuality, ...
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
