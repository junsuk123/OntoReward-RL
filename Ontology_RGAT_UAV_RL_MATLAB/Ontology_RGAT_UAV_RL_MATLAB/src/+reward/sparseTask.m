function r = sparseTask(status,cfg)
%SPARSETASK Binary task reward: the PBRS base that shaping must not alter.
r=cfg.reward.sparse.time*cfg.sim.dt;
switch status
    case 'success'
        r=r+cfg.reward.sparse.success;
    case {'unsafe_touchdown','flight_failure'}
        r=r+cfg.reward.sparse.failure;
    case 'timeout'
        r=r+cfg.reward.sparse.timeout;
end
end
