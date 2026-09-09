function [agent,H] = trainPPO(rewardMode,rgatModel,cfg)
%TRAINPPO Custom continuous-action PPO with multi-episode rollout batching.
% Episodes are accumulated into a buffer of at least cfg.ppo.rolloutSteps
% transitions before each update. Updating from one short episode made the
% minibatch larger than the batch, collapsing the inner loop to a single
% full-batch step per epoch and starving the policy of gradient steps.
agent=training.initPPO(cfg); optA=struct(); optC=struct(); stepA=0; stepC=0;
E=cfg.ppo.trainEpisodes;
H.return=zeros(E,1); H.success=zeros(E,1); H.length=zeros(E,1);
H.actorLoss=nan(E,1); H.criticLoss=nan(E,1); H.entropy=nan(E,1);
H.logStdExp=nan(E,1); H.iteration=zeros(E,1); H.status=strings(E,1);
if cfg.viz.training
    mon=viz.TrainingMonitor(sprintf('PPO - %s',rewardMode),rewardMode,cfg);
else
    mon=[];
end
ep=0; iter=0;
while ep<E
    iter=iter+1; epStart=ep+1;
    Ob={}; Ub={}; Ab={}; LPb={}; ADVb={}; RETb={}; nBuf=0;
    while nBuf<cfg.ppo.rolloutSteps && ep<E
        ep=ep+1;
        tr=collectEpisode(agent,rewardMode,rgatModel,20000+ep,cfg);
        [adv,ret]=training.computeGAE(tr.R,tr.V,tr.Dn,tr.lastV,cfg);
        Ob{end+1}=tr.O; Ub{end+1}=tr.U; Ab{end+1}=tr.A; %#ok<AGROW>
        LPb{end+1}=tr.LP; ADVb{end+1}=adv; RETb{end+1}=ret; %#ok<AGROW>
        nBuf=nBuf+numel(tr.R);
        H.return(ep)=sum(tr.R); H.success(ep)=double(strcmp(tr.status,'success'));
        H.length(ep)=numel(tr.R); H.status(ep)=string(tr.status); H.iteration(ep)=iter;
    end
    O=[Ob{:}]; U=[Ub{:}]; A=[Ab{:}]; LP=[LPb{:}]; adv=[ADVb{:}]; ret=[RETb{:}];
    % Standardize advantages once over the whole rollout, not per episode.
    adv=(adv-mean(adv))/(std(adv)+1e-8);
    n=numel(adv); aLoss=[]; cLoss=[]; entL=[];
    for epoch=1:cfg.ppo.epochs
        ord=randperm(n);
        for s=1:cfg.ppo.minibatch:n
            id=ord(s:min(s+cfg.ppo.minibatch-1,n));
            obsDL=dlarray(O(:,id)); uDL=dlarray(U(:,id)); actDL=dlarray(A(:,id));
            oldLPDL=dlarray(LP(id)); advDL=dlarray(adv(id)); retDL=dlarray(ret(id));
            stepA=stepA+1;
            [la,Ga,en]=dlfeval(@training.ppoActorGradients,agent.actor,obsDL,uDL,actDL,oldLPDL,advDL,cfg);
            [agent.actor,optA]=training.adamStep(agent.actor,Ga,optA,cfg.ppo.actorLR,stepA,cfg.ppo.gradClip);
            agent.actor.logStd=dlarray(max(-3,min(0.5,extractdata(agent.actor.logStd))));
            stepC=stepC+1;
            [lc,Gc]=dlfeval(@training.ppoCriticGradients,agent.critic,obsDL,retDL,cfg);
            [agent.critic,optC]=training.adamStep(agent.critic,Gc,optC,cfg.ppo.criticLR,stepC,cfg.ppo.gradClip);
            aLoss(end+1)=double(extractdata(la)); %#ok<AGROW>
            cLoss(end+1)=double(extractdata(lc)); %#ok<AGROW>
            entL(end+1)=double(extractdata(en)); %#ok<AGROW>
        end
    end
    stdNow=mean(exp(double(extractdata(agent.actor.logStd))));
    ix=epStart:ep;
    H.actorLoss(ix)=mean(aLoss); H.criticLoss(ix)=mean(cLoss);
    H.entropy(ix)=mean(entL); H.logStdExp(ix)=stdNow;
    if cfg.viz.training, update(mon,H,ep); drawnow limitrate; end
    fprintf(['PPO %-8s iter %3d | ep %4d-%4d/%4d | buf %5d (%3d upd) | return %+8.2f | ' ...
             'succ %5.1f%% | steps %5.1f | La %+.4f Lc %8.4f | std %.3f\n'], ...
        rewardMode,iter,epStart,ep,E,n,numel(aLoss),mean(H.return(ix)), ...
        100*mean(H.success(ix)),mean(H.length(ix)),mean(aLoss),mean(cLoss),stdNow);
end
if cfg.viz.training, snapshot(mon,H,ep); end
end

function tr = collectEpisode(agent,rewardMode,rgatModel,seed,cfg)
%COLLECTEPISODE One stochastic on-policy rollout with value and log-prob traces.
env=sim.resetState(seed,cfg); T=cfg.sim.maxSteps;
O=zeros(cfg.rl.obsDim,T); U=zeros(cfg.rl.actDim,T); A=zeros(cfg.rl.actDim,T);
R=zeros(1,T); V=zeros(1,T); LP=zeros(1,T); Dn=zeros(1,T);
status='timeout';
for k=1:T
    cur=sim.getCurrent(env,cfg);
    [a,u,lp]=training.samplePolicy(agent.actor,cur.obs,false,cfg);
    val=double(extractdata(training.criticForward(agent.critic,cur.obs)));
    [env2,~,r,done,info]=sim.step(env,a,cur,rewardMode,rgatModel,cfg);
    O(:,k)=cur.obs; U(:,k)=u; A(:,k)=a; R(k)=r; V(k)=val; LP(k)=lp; Dn(k)=done;
    env=env2; status=info.status;
    if done, break; end
end
if Dn(k)
    lastV=0;
else
    c=sim.getCurrent(env,cfg);
    lastV=double(extractdata(training.criticForward(agent.critic,c.obs)));
end
tr=struct('O',O(:,1:k),'U',U(:,1:k),'A',A(:,1:k),'R',R(1:k),'V',V(1:k), ...
          'LP',LP(1:k),'Dn',Dn(1:k),'lastV',lastV,'status',status);
end
