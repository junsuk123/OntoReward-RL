function [x,comp] = sampleMixtureProposal(muMotion,gpsXY,lidarPose,policy,cfg)
%SAMPLEMIXTUREPROPOSAL Sample motion/GNSS/LiDAR/broad mixture proposal.

N=size(muMotion,1);
cum=cumsum(policy.lambda);
u=rand(N,1);
comp=ones(N,1);
for k=1:4
    if k==1
        comp(u<=cum(k))=k;
    else
        comp(u>cum(k-1) & u<=cum(k))=k;
    end
end

x=muMotion;

m = comp==1;
x(m,:) = addDiagNoise(muMotion(m,:),policy.processSigma);

m = comp==2;
if any(m)
    mu = [repmat(gpsXY,sum(m),1),muMotion(m,3)];
    sig = [policy.sigmaGps,policy.sigmaGps,deg2rad(18)];
    x(m,:) = addDiagNoise(mu,sig);
end

m = comp==3;
if any(m)
    mu = repmat(lidarPose,sum(m),1);
    sig = [policy.sigmaLidar,policy.sigmaLidar,policy.sigmaLidarYaw];
    x(m,:) = addDiagNoise(mu,sig);
end

m = comp==4;
if any(m)
    sig=[cfg.pf.broadSigmaXY,cfg.pf.broadSigmaXY,cfg.pf.broadSigmaYaw];
    x(m,:) = addDiagNoise(muMotion(m,:),sig);
end

x(:,3)=wrapAngle(x(:,3));
end

function x = addDiagNoise(mu,sig)
x=mu+randn(size(mu)).*sig;
x(:,3)=wrapAngle(x(:,3));
end
