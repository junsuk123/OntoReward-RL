function [a,u,logp,mu] = samplePolicy(A,obs,deterministic,cfg) %#ok<INUSD>
%SAMPLEPOLICY Tanh-squashed Gaussian action and corrected log probability.
[muDL,stdDL]=training.actorForward(A,obs);
mu=double(extractdata(muDL)); stdv=double(extractdata(stdDL));
if deterministic
    u=mu;
else
    u=mu+stdv.*randn(size(mu));
end
a=tanh(u);
logp=sum(-0.5*((u-mu)./stdv).^2-log(stdv)-0.5*log(2*pi)-log(1-a.^2+1e-6),1);
end
