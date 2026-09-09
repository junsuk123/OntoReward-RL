function [env2,next,r,done,info] = step(env,action,cur,rewardMode,rgatModel,cfg)
omegaCmd=control.actionToRotorCmd(action,env.x,cfg);
x2=dynamics.rk4Step(env.x,omegaCmd,env.t,cfg);
t2=env.t+cfg.sim.dt;
% Ground contact: do not integrate through the floor.
if x2(3)<cfg.sim.groundZ, x2(3)=cfg.sim.groundZ; end
d2=dynamics.diagnostics(x2,omegaCmd,t2,cfg);
env2=env; env2.x=x2; env2.t=t2; env2.step=env.step+1; env2.prevSem=cur.sem; env2.lastDiag=d2;
next=sim.getCurrent(env2,cfg);
[done,status,viol]=sim.terminalStatus(x2,cfg);
if ~done && env2.step>=cfg.sim.maxSteps
    done=true; status='timeout';
end
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
info=struct('status',status,'omegaCmd',omegaCmd,'rewardParts',parts,'diag',d2,'viol',viol);
end
