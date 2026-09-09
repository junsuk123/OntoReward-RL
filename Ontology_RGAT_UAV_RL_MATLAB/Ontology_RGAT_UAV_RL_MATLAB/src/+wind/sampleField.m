function wI = sampleField(posI,t,cfg)
%SAMPLEFIELD Deterministic spatially and temporally varying 3-D wind [m/s].
% The field is the sum of mean shear, Fourier turbulence, local gust packets,
% and a finite-core vertical vortex. Each body panel samples this function at
% its own inertial position, so aerodynamic loading is distributed.
x=posI(:); z=max(x(3),0.05);
meanDir=cfg.wind.meanRef(:); horiz=meanDir; horiz(3)=0;
scale=((z+cfg.wind.z0)/(cfg.wind.zRef+cfg.wind.z0))^cfg.wind.shearAlpha;
wI=scale*horiz + [0;0;cfg.wind.verticalMean];
K=cfg.wind.modes.K; D=cfg.wind.modes.D;
for m=1:size(K,2)
    ph=dot(K(:,m),x)-cfg.wind.modes.omega(m)*t+cfg.wind.modes.phase(m);
    wI=wI + cfg.wind.modes.amp(m)*D(:,m)*sin(ph);
end
for k=1:numel(cfg.wind.gusts)
    g=cfg.wind.gusts(k); d=(x-g.center(:))./g.sigmaXYZ(:);
    space=exp(-0.5*dot(d,d)); time=exp(-0.5*((t-g.t0)/g.sigmaT)^2);
    wI=wI + g.vector(:)*space*time;
end
if cfg.wind.vortex.enable
    c=cfg.wind.vortex.center(:); rxy=x(1:2)-c(1:2); r=norm(rxy);
    if r>1e-6
        tang=[-rxy(2);rxy(1)]/r;
        rc=cfg.wind.vortex.coreRadius;
        speed=cfg.wind.vortex.strength*r/(r^2+rc^2);
        zgate=exp(-0.5*((x(3)-c(3))/(1.5*rc))^2);
        wI(1:2)=wI(1:2)+speed*zgate*tang;
    end
end
end
