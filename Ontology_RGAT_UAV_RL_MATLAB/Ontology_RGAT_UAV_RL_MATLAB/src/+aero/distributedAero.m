function out = distributedAero(x,t,cfg)
%DISTRIBUTEDAERO Surface-panel pressure drag + tangential skin friction.
p=x(1:3); vI=x(4:6); q=x(7:10); omegaB=x(11:13);
R=mathx.quatToRotm(q); P=cfg.aero.panels; nP=P.count;
Fsum=zeros(3,1); Msum=zeros(3,1); winds=zeros(3,nP); panelF=zeros(3,nP); Re=zeros(1,nP);
for i=1:nP
    rB=P.centerB(:,i); posI=p+R*rB;
    wI=wind.sampleField(posI,t,cfg); winds(:,i)=wI;
    vPointI=vI + R*cross(omegaB,rB);
    vRelB=R'*(wI-vPointI); % air velocity relative to panel
    nB=P.normalB(:,i); vn=dot(vRelB,nB); vt=vRelB-vn*nB;
    atm=aero.airProperties(posI(3),cfg);
    speed=max(norm(vRelB),1e-9);
    Rei=max(cfg.aero.minRe,atm.rho*speed*P.charLength(i)/atm.mu); Re(i)=Rei;
    if Rei < cfg.aero.transitionRe
        Cf=min(0.08,1.328/sqrt(Rei));
    else
        Cf=max(0.001,0.074/(Rei^0.2)-1742/Rei);
    end
    Fn=0.5*atm.rho*cfg.aero.CdNormal*P.area(i)*vn*abs(vn)*nB;
    vtn=norm(vt);
    if vtn>1e-10
        Ft=0.5*atm.rho*Cf*P.area(i)*vtn*vt;
    else
        Ft=zeros(3,1);
    end
    Fi=Fn+Ft;
    panelF(:,i)=Fi; Fsum=Fsum+Fi; Msum=Msum+cross(rB,Fi);
end
out.Fb=Fsum; out.Mb=Msum; out.panelForceB=panelF; out.panelWindI=winds;
out.meanWindI=mean(winds,2); out.stdWind=std(winds,0,2); out.Re=Re;
end
