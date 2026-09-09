function o=makeObservation(meas,s,cfg)
%MAKEOBSERVATION Normalize measured state + semantic context for PPO.
%   Position and velocity are pad-relative. The deck's own velocity is given
%   separately so the policy can feed it forward instead of having to infer a
%   moving target from the error signal alone, and the two energy channels let
%   it trade precision against reserve.
padVel=[0;0];
if isfield(meas,'padVelocity') && numel(meas.padVelocity)>=2
    padVel=meas.padVelocity(1:2);
end
o=[meas.pos(1:2)/4; meas.pos(3)/6; meas.vel/3; ...
   meas.rpy(1:2)/cfg.rl.maxRollPitch; mathx.wrapPi(meas.rpy(3))/pi; ...
   meas.omega/deg2rad(180); s.windRisk; s.visualStability; s.alignment; ...
   padVel(:)/max(cfg.semantic.padSpeedScale,1e-6); ...
   s.padMotion; s.batteryReserve; s.energyMargin];
o=max(-3,min(3,o));
if numel(o)~=cfg.rl.obsDim
    error('Observation dimension mismatch: built %d, cfg.rl.obsDim is %d.', ...
        numel(o),cfg.rl.obsDim);
end
end
