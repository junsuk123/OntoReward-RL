function [x,d]=stateToModel(s,cfg)
%STATETOMODEL Convert gateway ENU/FLU sample to the legacy algorithm contract.
%   x(1:3) and x(4:6) are pad-relative: the target rides a ground vehicle, so
%   the state the algorithms reason about is the state relative to the deck.
%   The world pose PX4 estimates is carried in the diagnostics for telemetry.
q=mathx.quatNormalize(s.quaternion_wxyz);
x=[s.position;s.velocity;q;s.angular_velocity;zeros(4,1)];
d=struct();
d.meanWindI=s.wind;
d.aeroForceMag=norm(s.aero_force);
d.aeroForceI=s.aero_force;
d.inducedPower=0;
d.markerQuality=s.marker_quality;
d.accelerationI=s.acceleration;
d.source='PX4/Isaac';
d.pad=padOf(s);
d.padPositionI=d.pad.position;
d.padVelocityI=d.pad.velocity;
d.padSpeed=norm(d.pad.velocity(1:2));
d.battery=batteryOf(s);
d.worldPositionI=worldVector(s,'position',s.position);
d.worldVelocityI=worldVector(s,'velocity',s.velocity);
% Keep the existing monitor contract. Isaac reports the resultant load, so
% visualization distributes it uniformly over the legacy panel locations.
R=mathx.quatToRotm(q);
panelForceB=(R.'*s.aero_force)/cfg.aero.panels.count;
d.aero=struct('panelForceB',repmat(panelForceB,1,cfg.aero.panels.count));
if numel(x)~=17, error('External state dimension mismatch.'); end
end

function p = padOf(s)
p=struct('valid',false,'source','static','position',[0;0;0],'velocity',[0;0;0], ...
    'yaw',0,'yawRate',0,'speed',0);
if ~isfield(s,'pad') || ~isstruct(s.pad), return; end
p.valid=logical(getfielddef(s.pad,'valid',false));
p.source=char(getfielddef(s.pad,'source','static'));
p.position=column(getfielddef(s.pad,'position',[0;0;0]),3);
p.velocity=column(getfielddef(s.pad,'velocity',[0;0;0]),3);
p.yaw=double(getfielddef(s.pad,'yaw',0));
p.yawRate=double(getfielddef(s.pad,'yaw_rate',0));
p.speed=double(getfielddef(s.pad,'speed',norm(p.velocity(1:2))));
end

function b = batteryOf(s)
%BATTERYOF The pack, as the gateway reports it. Field names are camel-cased so
%   the learning side never has to know the wire format.
b=struct('enabled',false,'source','unavailable','remainingJ',0,'initialJ',0, ...
    'capacityJ',0,'energyUsedJ',0,'powerW',0,'hoverPowerW',0, ...
    'hoverSecondsRemaining',0,'reserve',1,'landingReserveS',0, ...
    'stateOfCharge',1,'voltageV',0,'depleted',false);
if ~isfield(s,'battery') || ~isstruct(s.battery), return; end
w=s.battery;
b.enabled=logical(getfielddef(w,'enabled',false));
b.source=char(getfielddef(w,'source','unavailable'));
b.remainingJ=double(getfielddef(w,'remaining_j',0));
b.initialJ=double(getfielddef(w,'initial_j',0));
b.capacityJ=double(getfielddef(w,'capacity_j',0));
b.energyUsedJ=double(getfielddef(w,'energy_used_j',0));
b.powerW=double(getfielddef(w,'power_w',0));
b.hoverPowerW=double(getfielddef(w,'hover_power_w',0));
b.hoverSecondsRemaining=double(getfielddef(w,'hover_seconds_remaining',0));
b.reserve=double(getfielddef(w,'reserve',1));
b.landingReserveS=double(getfielddef(w,'landing_reserve_s',0));
b.stateOfCharge=double(getfielddef(w,'state_of_charge',1));
b.voltageV=double(getfielddef(w,'voltage_v',0));
b.depleted=logical(getfielddef(w,'depleted',false));
end

function v = worldVector(s,name,fallback)
v=column(fallback,3);
if isfield(s,'world') && isstruct(s.world) && isfield(s.world,name)
    v=column(s.world.(name),3);
end
end

function v = column(value,n)
v=double(value(:));
if numel(v)~=n, v=zeros(n,1); end
end

function v = getfielddef(st,name,default)
if isstruct(st) && isfield(st,name) && ~isempty(st.(name))
    v=st.(name);
else
    v=default;
end
end
