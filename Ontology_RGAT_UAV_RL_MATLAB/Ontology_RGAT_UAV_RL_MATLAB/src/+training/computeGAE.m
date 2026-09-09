function [adv,ret] = computeGAE(r,v,done,lastV,cfg)
%COMPUTEGAE Generalized advantage estimation for one episode segment.
% Advantages are returned unnormalized: trainPPO standardizes them once over
% the whole multi-episode rollout, which is the correct scope.
T=numel(r); adv=zeros(1,T); gae=0;
for t=T:-1:1
    if t==T, vn=lastV; dn=done(t); else, vn=v(t+1); dn=done(t); end
    delta=r(t)+cfg.ppo.gamma*(1-dn)*vn-v(t);
    gae=delta+cfg.ppo.gamma*cfg.ppo.lambdaGAE*(1-dn)*gae;
    adv(t)=gae;
end
ret=adv+v;
end
