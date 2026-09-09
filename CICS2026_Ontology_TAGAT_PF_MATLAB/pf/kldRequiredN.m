function Nreq = kldRequiredN(p,cfg)
%KLDREQUIREDN Fox-style occupied-bin KLD particle-count approximation.

bx=floor(p(:,1)/cfg.pf.kldBinXY);
by=floor(p(:,2)/cfg.pf.kldBinXY);
ba=floor(wrapAngle(p(:,3))/cfg.pf.kldBinYaw);
k=size(unique([bx by ba],"rows"),1);

if k<=1
    Nreq=cfg.pf.minN;
    return;
end

epsK=cfg.pf.kldEpsilon;
delta=cfg.pf.kldDelta;
z=sqrt(2)*erfinv(2*(1-delta)-1);

term=1-2/(9*(k-1))+z*sqrt(2/(9*(k-1)));
Nreq=ceil((k-1)/(2*epsK)*term^3);
Nreq=clamp(Nreq,cfg.pf.minN,cfg.pf.maxN);
end
