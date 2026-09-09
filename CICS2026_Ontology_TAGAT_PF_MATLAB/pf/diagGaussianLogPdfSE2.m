function lp = diagGaussianLogPdfSE2(x,mu,sig)
%DIAGGAUSSIANLOGPDFSE2 Diagonal Gaussian with wrapped yaw residual.

if size(mu,1)==1
    mu=repmat(mu,size(x,1),1);
end
e=x-mu;
e(:,3)=wrapAngle(e(:,3));
v=sig.^2;
lp=-0.5*sum((e.^2)./v,2)-sum(log(sig))-1.5*log(2*pi);
end
