function pack = buildGraphDataset(data,trainIdx,cfg) %#ok<INUSD>
%BUILDGRAPHDATASET Convert schema table to node-wise graph features.
%
% X has dimensions [featureChannels x nodes x time].
% Latent output nodes have no direct measurements.

Ftab = data.features;
obs = [ ...
    Ftab.GPSFix, Ftab.NumSV, Ftab.GPSInnovation, Ftab.GPSSpeedResidual, ...
    Ftab.LidarValidRatio, Ftab.LidarInlierRatio, Ftab.LidarICPRMSE, Ftab.LidarStructure, ...
    Ftab.OdomCovXY, Ftab.OdomYawVar, Ftab.Dynamics, Ftab.OpenSkyScore];

T = size(obs,1);
N = 16;

mu = mean(obs(trainIdx,:),1,"omitnan");
sd = std(obs(trainIdx,:),0,1,"omitnan");
sd(sd<1e-6 | ~isfinite(sd)) = 1;

z = (obs-mu)./sd;
z(~isfinite(z)) = 0;

dz = [zeros(1,size(z,2)); diff(z,1,1)];
dz = clamp(dz,-5,5);

X = zeros(5,N,T,"single");
for n=1:12
    X(1,n,:) = single(z(:,n));
    X(2,n,:) = single(dz(:,n));
    X(3,n,:) = single(~isfinite(obs(:,n)));
    X(4,n,:) = single(abs(z(:,n)));
    X(5,n,:) = 1;
end

% Latent ontology nodes: missing flag + bias only.
for n=13:16
    X(1,n,:) = 0;
    X(2,n,:) = 0;
    X(3,n,:) = 1;
    X(4,n,:) = 0;
    X(5,n,:) = 1;
end

Y = single([ ...
    data.targets.GNSSReliability.'; ...
    data.targets.LiDARReliability.'; ...
    data.targets.OdomReliability.'; ...
    data.targets.Difficulty.']);

pack.X = X;
pack.Y = Y;
pack.mu = mu;
pack.sd = sd;
pack.nodeValueNames = string(Ftab.Properties.VariableNames);
end
