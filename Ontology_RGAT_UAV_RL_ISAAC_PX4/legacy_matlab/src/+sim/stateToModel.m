function [x,d]=stateToModel(s,cfg)
%STATETOMODEL Convert gateway ENU/FLU sample to the legacy algorithm contract.
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
% Keep the existing monitor contract. Isaac reports the resultant load, so
% visualization distributes it uniformly over the legacy panel locations.
R=mathx.quatToRotm(q);
panelForceB=(R.'*s.aero_force)/cfg.aero.panels.count;
d.aero=struct('panelForceB',repmat(panelForceB,1,cfg.aero.panels.count));
if numel(x)~=17, error('External state dimension mismatch.'); end
end
