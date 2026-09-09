function r = sparseTask(status,cfg)
%SPARSETASK Binary task reward: the PBRS base that shaping must not alter.
%   External version: running out of energy in the air is a distinct failure,
%   so it gets its own terminal value instead of being scored as a timeout.
r=cfg.reward.sparse.time*cfg.sim.dt;
switch status
    case 'success'
        r=r+cfg.reward.sparse.success;
    case {'unsafe_touchdown','flight_failure'}
        r=r+cfg.reward.sparse.failure;
    case 'battery_depleted'
        r=r+cfg.reward.sparse.batteryDepleted;
    case 'timeout'
        r=r+cfg.reward.sparse.timeout;
end
end
