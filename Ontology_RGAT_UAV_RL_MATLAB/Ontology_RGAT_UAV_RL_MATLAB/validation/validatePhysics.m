function report = validatePhysics(cfg)
%VALIDATEPHYSICS Deterministic sanity checks of dynamics and aero modules.
report=struct();

% 1) Quaternion rotation identity.
R=mathx.quatToRotm([1;0;0;0]); report.identityRotationError=norm(R-eye(3),'fro');
assert(report.identityRotationError<1e-12,'Quaternion identity rotation failed.');

% 2) Zero relative air speed -> near-zero aero load.
c0=cfg; c0.wind.meanRef=[0;0;0]; c0.wind.modes.amp(:)=0; c0.wind.gusts=struct([]); c0.wind.vortex.enable=false;
omegaHover=sqrt(c0.drone.mass*c0.sim.g/(4*c0.prop.kT));
x=[0;0;2; 0;0;0; 1;0;0;0; 0;0;0; omegaHover*ones(4,1)];
a=aero.distributedAero(x,0,c0); report.zeroAeroForce=norm(a.Fb); report.zeroAeroMoment=norm(a.Mb);
assert(report.zeroAeroForce<1e-10,'Zero-flow aero force is not zero.');

% 3) Hover equilibrium without wind/ground effect.
c1=c0; c1.prop.groundEffect.enable=false;
cmd=omegaHover*ones(4,1); [dx,~]=dynamics.eom(x,cmd,0,c1); report.hoverAccel=norm(dx(4:6)); report.hoverAngularAccel=norm(dx(11:13));
assert(report.hoverAccel<2e-3,'Hover translational equilibrium failed.');
assert(report.hoverAngularAccel<2e-6,'Hover rotational equilibrium failed.');

% 4) Positive x wind should produce positive x drag on stationary body.
c2=c0; c2.wind.meanRef=[4;0;0]; a2=aero.distributedAero(x,0,c2); report.dragForceX=a2.Fb(1);
assert(report.dragForceX>0,'Panel drag direction check failed.');

% 5) Mixer recovers approximately hover thrust.
w=prop.mixWrenchToOmega(c1.drone.mass*c1.sim.g,[0;0;0],c1); rr=prop.rotorForces(w,2,c1);
report.mixerThrustError=abs(sum(rr.thrust)-c1.drone.mass*c1.sim.g);
assert(report.mixerThrustError<1e-6,'Mixer hover thrust check failed.');

% 6) Surface load is genuinely spatially varying under configured field.
a3=aero.distributedAero(x,5.0,cfg); report.panelWindStdNorm=norm(std(a3.panelWindI,0,2));
assert(report.panelWindStdNorm>1e-4,'Configured wind field is not spatially varying across panels.');

report.pass=true;
fprintf('Physics validation PASS\n');
fprintf('  hover accel       : %.3e m/s^2\n',report.hoverAccel);
fprintf('  zero-flow aero F  : %.3e N\n',report.zeroAeroForce);
fprintf('  +x wind drag Fx   : %.3f N\n',report.dragForceX);
fprintf('  panel wind std    : %.3e m/s\n',report.panelWindStdNorm);
end
