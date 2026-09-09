function [env2,next,r,done,info]=step(env,action,cur,rewardMode,rgatModel,cfg)
s=env.bridge.step(action);
[x2,d2]=sim.stateToModel(s,cfg);
% Advance the episode clock by the simulated time that actually elapsed. The
% bridge paces to cfg.sim.dt but can only return a sample on a PX4 publication
% boundary, so assuming the nominal period leaves log.t short of the truth.
env2=env; env2.x=x2; env2.t=env.t+elapsedSeconds(env,s,cfg); env2.step=env.step+1;
env2.prevSem=cur.sem; env2.lastDiag=d2; env2.externalState=s;
env2.hasBeenAirborne=env.hasBeenAirborne || ~logical(s.landed);
next=sim.getCurrent(env2,cfg);
[done,status,viol]=sim.terminalStatus(x2,cfg,env2.hasBeenAirborne);
if ~done && env2.hasBeenAirborne && logical(s.landed)
    done=true;
    if viol<=1, status='success'; else, status='unsafe_touchdown'; end
end
if ~done && env2.step>=cfg.sim.maxSteps, done=true; status='timeout'; end
switch lower(rewardMode)
    case 'manual'
        r=reward.manualDense(cur,next,action,status,viol,cfg); parts=struct();
    case 'proposed'
        [r,parts]=reward.proposedPBRS(cur,next,status,rgatModel,cfg);
    case 'sparse'
        r=reward.sparseTask(status,cfg); parts=struct();
    otherwise
        error('Unknown reward mode: %s',rewardMode);
end
env2.done=done;
info=struct('status',status,'rewardParts',parts,'diag',d2,'viol',viol, ...
    'armed',logical(s.armed),'landed',logical(s.landed),'navState',s.nav_state);
if done && strcmpi(cfg.external.target,'sitl')
    try
        env2.bridge.disarm();
    catch
    end
end
end

function dt=elapsedSeconds(env,s,cfg)
dt=cfg.sim.dt;
if ~isfield(s,'px4_time_us') || ~isfield(env,'externalState') || ...
        ~isfield(env.externalState,'px4_time_us')
    return;
end
measured=(double(s.px4_time_us)-double(env.externalState.px4_time_us))*1e-6;
% Ignore a clock reset or a stalled sample; the nominal period is the sane
% fallback and the pacing loop already guards against a stalled simulator.
if isfinite(measured) && measured>0 && measured<10*cfg.sim.dt
    dt=measured;
end
end
