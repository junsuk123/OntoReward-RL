function agent = initPPO(cfg)
%INITPPO Initialize actor and critic parameter structs for custom PPO.
rng(cfg.seed+202,'twister');
h=cfg.ppo.hidden; od=cfg.rl.obsDim; ad=cfg.rl.actDim; s=0.10;
A.W1=dlarray(s*randn(h,od)); A.b1=dlarray(zeros(h,1));
A.W2=dlarray(s*randn(h,h));  A.b2=dlarray(zeros(h,1));
A.Wm=dlarray(s*randn(ad,h)); A.bm=dlarray(zeros(ad,1));
A.logStd=dlarray(cfg.ppo.initLogStd*ones(ad,1));
C.W1=dlarray(s*randn(h,od)); C.b1=dlarray(zeros(h,1));
C.W2=dlarray(s*randn(h,h));  C.b2=dlarray(zeros(h,1));
C.Wv=dlarray(s*randn(1,h));  C.bv=dlarray(0);
agent=struct('actor',A,'critic',C);
end
