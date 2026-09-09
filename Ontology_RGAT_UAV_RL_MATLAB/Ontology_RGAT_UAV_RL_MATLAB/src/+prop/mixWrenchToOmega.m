function omegaCmd = mixWrenchToOmega(totalThrust,momentB,cfg)
%MIXWRENCHTOOMEGA Allocate desired wrench to four squared rotor speeds.
B=zeros(4,4);
for i=1:4
    r=cfg.prop.rotorPosB(:,i);
    B(1,i)=cfg.prop.kT;
    tau=cross(r,[0;0;cfg.prop.kT]);
    B(2,i)=tau(1); B(3,i)=tau(2); B(4,i)=cfg.prop.spinDir(i)*cfg.prop.kQ;
end
u=B\[totalThrust;momentB(:)];
u=max(u,0);
omegaCmd=sqrt(u);
omegaCmd=max(cfg.prop.omegaMin,min(cfg.prop.omegaMax,omegaCmd));
end
