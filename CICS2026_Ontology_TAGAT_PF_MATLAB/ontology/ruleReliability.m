function rel = ruleReliability(data,cfg) %#ok<INUSD>
%RULERELIABILITY Smooth interpretable ontology/rule baseline.
% No hard final decision is made; outputs are bounded soft reliabilities.

F = data.features;

gpsAvail = double(F.GPSFix>=2);
sv = clamp(F.NumSV/10,0,1);
innov = exp(-F.GPSInnovation/8);
speedCons = exp(-F.GPSSpeedResidual/2.5);
rG = gpsAvail .* clamp(0.25 + 0.25*sv + 0.32*innov + 0.18*speedCons,0,1);

valid = clamp(F.LidarValidRatio,0,1);
inlier = clamp(F.LidarInlierRatio,0,1);
rmse = exp(-F.LidarICPRMSE/0.8);
structure = clamp(F.LidarStructure,0,1);
rL = clamp(0.25*valid + 0.35*inlier + 0.25*rmse + 0.15*structure,0,1);

covScore = exp(-0.8*sqrt(max(F.OdomCovXY,0)));
yawScore = exp(-2.0*sqrt(max(F.OdomYawVar,0)));
dynScore = exp(-0.35*F.Dynamics);
rO = clamp(0.45*covScore + 0.20*yawScore + 0.35*dynScore,0,1);

difficulty = 1 - (0.55*max([rG,rL,rO],[],2)+0.45*mean([rG,rL,rO],2));
difficulty = clamp(difficulty + 0.10*clamp(F.Dynamics,0,1),0,1);

rel.Y = [rG.';rL.';rO.';difficulty.'];
rel.edgeAttention = [];
rel.mode = "RULE";
end
