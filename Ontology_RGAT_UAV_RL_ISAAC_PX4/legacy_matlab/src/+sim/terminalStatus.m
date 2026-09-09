function [done,status,viol]=terminalStatus(x,cfg,hasBeenAirborne,batteryDepleted)
%TERMINALSTATUS Episode outcome for one external state sample.
% HASBEENAIRBORNE guards the ground test: with a real flight stack the vehicle
% can still be sitting on the pad when control is handed over, and that is a
% start condition, not a touchdown.
%
% The state is pad-relative, so the arena checks measure the distance to the
% deck: a rover that drives away faster than the drone follows it is a lost
% target, which is a mission failure and not a crash.
%
% VIOL gains a fifth ratio. Touching down on a deck the vehicle does not share
% a horizontal velocity with tips the airframe over, so closing speed is a
% landing criterion. It applies to the static control condition as well, which
% is why numbers here are not comparable with the older fixed-pad runs that had
% no such criterion.
if nargin<3, hasBeenAirborne=true; end
if nargin<4, batteryDepleted=false; end
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10));
rate=norm(x(11:13)); tilt=norm(rpy(1:2));
relSpeedXY=norm(v(1:2));
viol=max([norm(p(1:2))/cfg.criteria.xy,abs(v(3))/cfg.criteria.vz, ...
    tilt/cfg.criteria.tilt,rate/cfg.criteria.rate, ...
    relSpeedXY/cfg.criteria.relSpeedXY]);
done=false; status='running';
if hasBeenAirborne && p(3)<=cfg.sim.groundZ
    done=true;
    if viol<=1, status='success'; else, status='unsafe_touchdown'; end
elseif batteryDepleted
    % Out of energy in the air. Not a crash yet, but the mission is over and
    % the vehicle is about to become one, so it is its own failure mode rather
    % than a timeout.
    done=true; status='battery_depleted';
elseif norm(p(1:2))>cfg.sim.worldXYLimit || p(3)>cfg.sim.maxAltitude || ...
        tilt>cfg.sim.crashTilt
    done=true; status='flight_failure';
end
end
