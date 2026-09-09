function s = computeFeatures(x,diagOut,meas,prev,cfg)
%COMPUTEFEATURES Semantic states following the landing-paper concepts.
%   External version. Two facts the in-process simulator did not have change
%   what "landing safely" means, so they become semantic channels rather than
%   being smuggled into the observation vector alone:
%
%     PadMotion       the pad rides a ground vehicle, so the target moves and
%                     the horizontal velocity still to be cancelled is part of
%                     the state rather than a nuisance.
%     BatteryReserve  the pack starts each episode nearly empty, so whether
%                     there is still energy to finish a descent competes with
%                     doing it precisely.
%
%   Every position and velocity in MEAS is already pad-relative; the gateway
%   expresses them that way (docs/ARCHITECTURE.md, "Coordinates").
posErr=norm(meas.pos(1:2));
vz=meas.vel(3); tilt=norm(meas.rpy(1:2)); rate=norm(meas.omega);
closingSpeed=norm(meas.vel(1:2));      % pad-relative, so this is what must go to zero
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

%% Pad motion
% Two independent contributions, both saturating in [0,1]: how hard the deck is
% driving, and how far the vehicle currently is from being able to touch down on
% it. A stationary deck with a matched vehicle gives exactly zero.
relTol=max(cfg.criteria.relSpeedXY,1e-6);
padSpeed=norm(padVelocity(diagOut));
padSpeedRatio=min(1,padSpeed/max(cfg.semantic.padSpeedScale,1e-6));
closingRatio=min(1,closingSpeed/relTol);
padMotion=min(1,0.5*padSpeedRatio+0.5*closingRatio);

%% Energy
% hoverSecondsRemaining is the pack's own currency: how long it could hold a
% hover. The margin is what is left after paying for the descent still to fly
% plus the reserve the vehicle should touch down with, expressed in units of the
% episode horizon so it is comparable with the other normalized channels.
batt=batteryOf(diagOut);
if ~batt.enabled
    energyMargin=1; batteryReserve=1; energyNeeded=0;
else
    energyNeeded=max(meas.pos(3),0)/max(cfg.battery.planDescentRate,1e-6) ...
        +batt.landingReserveS;
    energyMargin=(batt.hoverSecondsRemaining-energyNeeded)/max(cfg.sim.maxTime,1e-6);
    energyMargin=max(-3,min(3,energyMargin));
    batteryReserve=min(1,max(0,0.5+0.5*energyMargin/max(cfg.semantic.energyScale,1e-6)));
end

%% Touchdown safety
% The original product, with the horizontal closing speed folded in: arriving on
% the deck with velocity it does not share tips the airframe over whether the
% deck is moving or not, so this applies to the static control condition too.
touchdownSafety=alignment*attitudeStability*visualStability ...
    *exp(-abs(vz)/cfg.semantic.vzSafeScale)*(1-windRisk) ...
    *exp(-closingSpeed/relTol)*batteryReserve;

s=struct('positionError',posErr,'verticalSpeed',vz,'tilt',tilt,'angularRate',rate, ...
    'windRisk',windRisk,'markerQuality',meas.markerQuality,'visualStability',visualStability, ...
    'alignment',alignment,'attitudeStability',attitudeStability,'touchdownSafety',touchdownSafety, ...
    'padMotion',padMotion,'batteryReserve',batteryReserve, ...
    'meanWind',meanWind,'windAccel',aWind,'windDirChange',dTheta, ...
    'rDrag',rDrag,'rAcc',rAcc,'rDir',rDir, ...
    'closingSpeed',closingSpeed,'padSpeed',padSpeed, ...
    'energyMargin',energyMargin,'energyNeededS',energyNeeded, ...
    'hoverSecondsRemaining',batt.hoverSecondsRemaining);
end

function v = padVelocity(diagOut)
v=[0;0;0];
if isfield(diagOut,'padVelocityI') && numel(diagOut.padVelocityI)>=2
    v=diagOut.padVelocityI(:);
end
v=v(1:2);
end

function b = batteryOf(diagOut)
%BATTERYOF Battery view with a safe default, so a link that reports no energy
%   (the MAVLink fallback) leaves every energy channel neutral instead of
%   reading as an empty pack.
b=struct('enabled',false,'hoverSecondsRemaining',0,'landingReserveS',0,'reserve',1);
if isfield(diagOut,'battery') && isstruct(diagOut.battery)
    f=fieldnames(diagOut.battery);
    for k=1:numel(f), b.(f{k})=diagOut.battery.(f{k}); end
end
end
