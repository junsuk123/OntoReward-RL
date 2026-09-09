function out = rotorForces(omegaRotor,z,cfg)
%ROTORFORCES Individual rotor thrust, moment and ideal induced-power proxy.
w=max(cfg.prop.omegaMin,min(cfg.prop.omegaMax,omegaRotor(:)));
T=cfg.prop.kT*w.^2;
T=T*prop.groundEffectGain(z,cfg);
F=zeros(3,1); M=zeros(3,1);
for i=1:4
    Fi=[0;0;T(i)]; F=F+Fi;
    M=M+cross(cfg.prop.rotorPosB(:,i),Fi)+[0;0;cfg.prop.spinDir(i)*cfg.prop.kQ*w(i)^2];
end
atm=aero.airProperties(max(z,0),cfg); A=pi*cfg.drone.rotorRadius^2;
Pind=sum((max(T,0).^(3/2))./sqrt(max(2*atm.rho*A,1e-9)));
out.Fb=F; out.Mb=M; out.thrust=T; out.omega=w; out.inducedPower=Pind;
end
