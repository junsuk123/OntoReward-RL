function env=resetState(seed,cfg)
%RESETSTATE Reset the episode and hand over once PX4 holds the entry pose.
%   A long sweep outlives one PX4 SITL session: after hours the simulated
%   battery status goes stale under lockstep and PX4 refuses to arm, which
%   surfaces here as a reset that never reaches the entry pose. When
%   run_pipeline owns the simulator, cycle it and try again rather than
%   throwing away the run.
attempts=1+resetRecoveries(cfg);
for k=1:attempts
    try
        env=attemptReset(seed,cfg);
        return;
    catch err
        theStack=stack.current();
        if k==attempts || isempty(theStack)
            rethrow(err);
        end
        warning('sim:resetRecovery', ...
            'Reset failed (%s). Restarting the simulator and retrying (%d of %d).', ...
            err.message,k,attempts-1);
        theStack.restart();
    end
end
end

function env=attemptReset(seed,cfg)
b=bridge.PX4Bridge(cfg);
try
    s=b.reset(seed);
catch err
    % Release the MATLAB-side UDP port so a retry can bind it again.
    delete(b);
    rethrow(err);
end
[x,diagOut]=sim.stateToModel(s,cfg);
env=struct('x',x,'t',0,'step',0,'prevSem',[],'lastDiag',diagOut, ...
    'done',false,'seed',seed,'bridge',b,'externalState',s, ...
    'hasBeenAirborne',~logical(s.landed));
end

function n=resetRecoveries(cfg)
n=0;
if isfield(cfg,'external') && isfield(cfg.external,'resetRecoveries')
    n=max(0,round(cfg.external.resetRecoveries));
end
end
