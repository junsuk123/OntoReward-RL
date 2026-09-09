function [done,status,viol] = terminalStatus(x,cfg)
%TERMINALSTATUS Episode termination test and normalized landing-criteria violation.
% viol is the worst of the four safety ratios: viol<=1 means every landing
% criterion is satisfied. It is also handed to the reward so a terminal penalty
% can be graded instead of being an information-free cliff.
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10)); rate=norm(x(11:13));
tilt=norm(rpy(1:2));
viol=max([norm(p(1:2))/cfg.criteria.xy, abs(v(3))/cfg.criteria.vz, ...
          tilt/cfg.criteria.tilt, rate/cfg.criteria.rate]);
done=false; status='running';
if p(3)<=cfg.sim.groundZ
    done=true;
    if viol<=1, status='success'; else, status='unsafe_touchdown'; end
elseif norm(p(1:2))>cfg.sim.worldXYLimit || p(3)>cfg.sim.maxAltitude || tilt>cfg.sim.crashTilt
    done=true; status='flight_failure';
end
end
