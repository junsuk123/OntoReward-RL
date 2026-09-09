function [dx,diagOut] = eom(x,omegaCmd,t,cfg)
%EOM 6-DOF rigid-body equations augmented with 4 first-order motor states.
% State: [pI(3); vI(3); q_BI scalar-first(4); omegaB(3); rotorOmega(4)]
p=x(1:3); v=x(4:6); q=x(7:10); omega=x(11:13); wr=x(14:17);
R=mathx.quatToRotm(q);
rot=prop.rotorForces(wr,p(3),cfg);
aer=aero.distributedAero(x,t,cfg);
Fb=rot.Fb+aer.Fb;
Mb=rot.Mb+aer.Mb;
gI=[0;0;-cfg.sim.g];
pdot=v;
vdot=R*Fb/cfg.drone.mass + gI;
Omega=[0 -omega'; omega -mathx.skew(omega)];
qdot=0.5*Omega*q;
omegadot=cfg.drone.J\(Mb-cross(omega,cfg.drone.J*omega));
wrdot=(omegaCmd(:)-wr)/cfg.prop.tauMotor;
dx=[pdot;vdot;qdot;omegadot;wrdot];
if nargout>1
    diagOut=struct('rotor',rot,'aero',aer,'Fb',Fb,'Mb',Mb,'accI',vdot);
end
end
