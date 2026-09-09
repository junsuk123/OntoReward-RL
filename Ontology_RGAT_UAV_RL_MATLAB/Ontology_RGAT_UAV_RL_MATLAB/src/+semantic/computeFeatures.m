function s = computeFeatures(x,diagOut,meas,prev,cfg)
%COMPUTEFEATURES Semantic states following the landing-paper concepts.
posErr=norm(meas.pos(1:2));
vz=meas.vel(3); tilt=norm(meas.rpy(1:2)); rate=norm(meas.omega);
Fcap=max(1.0,cfg.drone.maxTotalThrust*max(cos(tilt),0.1)-cfg.drone.mass*cfg.sim.g);
rDrag=min(1,diagOut.aeroForceMag/Fcap);
meanWind=diagOut.meanWindI;
if nargin<4 || isempty(prev) || ~isfield(prev,'meanWind')
    aWind=0; dTheta=0;
else
    aWind=norm(meanWind-prev.meanWind)/cfg.sim.dt;
    n1=norm(meanWind); n0=norm(prev.meanWind);
    if n1<1e-8 || n0<1e-8
        dTheta=0;
    else
        c=max(-1,min(1,dot(meanWind,prev.meanWind)/(n1*n0)));
        dTheta=acos(c);
    end
end
rAcc=min(1,aWind/cfg.semantic.windAccelThr);
rDir=min(1,dTheta/cfg.semantic.windDirThr);
z=cfg.semantic.windRiskW'*[rDrag;rAcc;rDir]+cfg.semantic.windRiskB;
windRisk=1/(1+exp(-z));
alignment=exp(-posErr/cfg.semantic.alignScale);
attitudeStability=exp(-tilt/cfg.semantic.attTiltScale-rate/cfg.semantic.attRateScale);
visualStability=max(0,min(1,meas.markerQuality));
touchdownSafety=alignment*attitudeStability*visualStability ...
    *exp(-abs(vz)/cfg.semantic.vzSafeScale)*(1-windRisk);
s=struct('positionError',posErr,'verticalSpeed',vz,'tilt',tilt,'angularRate',rate, ...
    'windRisk',windRisk,'markerQuality',meas.markerQuality,'visualStability',visualStability, ...
    'alignment',alignment,'attitudeStability',attitudeStability,'touchdownSafety',touchdownSafety, ...
    'meanWind',meanWind,'windAccel',aWind,'windDirChange',dTheta,'rDrag',rDrag,'rAcc',rAcc,'rDir',rDir);
end
