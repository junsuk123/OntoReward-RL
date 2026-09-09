function [done,status,viol]=terminalStatus(x,cfg,hasBeenAirborne)
%TERMINALSTATUS Episode outcome for one external state sample.
% HASBEENAIRBORNE guards the ground test: with a real flight stack the vehicle
% can still be sitting on the pad when control is handed over, and that is a
% start condition, not a touchdown.
if nargin<3, hasBeenAirborne=true; end
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10));
rate=norm(x(11:13)); tilt=norm(rpy(1:2));
viol=max([norm(p(1:2))/cfg.criteria.xy,abs(v(3))/cfg.criteria.vz, ...
    tilt/cfg.criteria.tilt,rate/cfg.criteria.rate]);
done=false; status='running';
if hasBeenAirborne && p(3)<=cfg.sim.groundZ
    done=true;
    if viol<=1, status='success'; else, status='unsafe_touchdown'; end
elseif norm(p(1:2))>cfg.sim.worldXYLimit || p(3)>cfg.sim.maxAltitude || ...
        tilt>cfg.sim.crashTilt
    done=true; status='flight_failure';
end
end

