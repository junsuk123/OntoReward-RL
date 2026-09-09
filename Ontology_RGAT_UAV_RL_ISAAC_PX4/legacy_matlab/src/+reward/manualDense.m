function r = manualDense(cur,next,action,status,viol,cfg)
%MANUALDENSE Hand-designed baseline reward: explicit arbitrary fixed weights.
%   External version. The moving deck and the energy budget get hand-tuned
%   weights here, which is the point of the baseline: the proposed arm has to
%   derive the same trade-offs from the ontology instead of being told them.
w=cfg.reward.manual;
r=-cfg.sim.dt*(w.wPos*min(cur.sem.positionError,3) ...
    +w.wVel*norm(cur.meas.vel) + w.wTilt*cur.sem.tilt ...
    +w.wRate*cur.sem.angularRate + w.wWind*cur.sem.windRisk ...
    +w.wPadTrack*min(cur.sem.closingSpeed,3) ...
    +w.wEnergy*(1-cur.sem.batteryReserve) ...
    +w.wAct*sum(action(:).^2) + w.time);
% Reward progress to reduce purely static penalties.
r=r+0.8*(cur.sem.positionError-next.sem.positionError);
switch status
    case 'success'
        r=r+w.success;
    case {'unsafe_touchdown','flight_failure'}
        % Graded rather than a cliff: a near miss costs a fraction of a hard
        % crash, so the policy gets a gradient toward safe touchdown instead of
        % an information-free binary penalty.
        grade=min(1,max(0,(viol-1)/w.violSpan));
        r=r+w.failure*(w.failureFloor+(1-w.failureFloor)*grade);
    case 'battery_depleted'
        r=r+w.batteryDepleted;
    case 'timeout'
        % Never touching down is a mission failure. Without this, hovering out
        % the clock strictly dominates attempting a landing.
        r=r+w.timeout;
end
end
