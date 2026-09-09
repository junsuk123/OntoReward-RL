function logq = mixtureProposalLogPdf(x,muMotion,gpsXY,lidarPose,policy,cfg)
%MIXTUREPROPOSALLOGPDF Numerically stable log density of four-component proposal.

N=size(x,1);
L=-inf(N,4);

if policy.lambda(1)>0
    L(:,1)=log(policy.lambda(1))+diagGaussianLogPdfSE2(x,muMotion,policy.processSigma);
end

if policy.lambda(2)>0 && all(isfinite(gpsXY))
    mu=[repmat(gpsXY,N,1),muMotion(:,3)];
    sig=[policy.sigmaGps,policy.sigmaGps,deg2rad(18)];
    L(:,2)=log(policy.lambda(2))+diagGaussianLogPdfSE2(x,mu,sig);
end

if policy.lambda(3)>0 && all(isfinite(lidarPose))
    mu=repmat(lidarPose,N,1);
    sig=[policy.sigmaLidar,policy.sigmaLidar,policy.sigmaLidarYaw];
    L(:,3)=log(policy.lambda(3))+diagGaussianLogPdfSE2(x,mu,sig);
end

if policy.lambda(4)>0
    sig=[cfg.pf.broadSigmaXY,cfg.pf.broadSigmaXY,cfg.pf.broadSigmaYaw];
    L(:,4)=log(policy.lambda(4))+diagGaussianLogPdfSE2(x,muMotion,sig);
end

mx=max(L,[],2);
logq=mx+log(sum(exp(L-mx),2)+realmin);
end
