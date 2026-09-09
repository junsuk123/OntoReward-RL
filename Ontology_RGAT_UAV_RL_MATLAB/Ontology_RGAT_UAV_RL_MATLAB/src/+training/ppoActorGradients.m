function [loss,G,entropyMean] = ppoActorGradients(A,obs,u,a,oldLogp,adv,cfg)
[mu,stdv]=training.actorForward(A,obs);
logp=sum(-0.5*((u-mu)./stdv).^2-log(stdv)-0.5*log(2*pi)-log(1-a.^2+1e-6),1);
ratio=exp(logp-oldLogp);
clipped=max(1-cfg.ppo.clip,min(1+cfg.ppo.clip,ratio));
s1=ratio.*adv; s2=clipped.*adv;
obj=min(s1,s2);
entropy=sum(log(stdv*sqrt(2*pi*exp(1))),1);
entropyMean=mean(entropy,'all');
loss=-mean(obj,'all')-cfg.ppo.entropyCoef*entropyMean;
[G.W1,G.b1,G.W2,G.b2,G.Wm,G.bm,G.logStd]=dlgradient(loss, ...
    A.W1,A.b1,A.W2,A.b2,A.Wm,A.bm,A.logStd);
end
