function D = generateRGATDataset(cfg)
%GENERATERGATDATASET Produce success/failure graph states from perturbed expert rollouts.
fprintf('Generating %d behavior episodes...\n',cfg.rgat.dataEpisodes);
Xs={}; ys=[]; meta=[]; template=[];
nEp=cfg.rgat.dataEpisodes;
S=struct('success',zeros(nEp,1),'noise',zeros(nEp,1),'samples',zeros(nEp,1));
if cfg.viz.training, mon=viz.DatasetMonitor(cfg); else, mon=[]; end
for ep=1:nEp
    lo=cfg.rgat.noiseRange(1); hi=cfg.rgat.noiseRange(2);
    severity=exp(log(lo)+(log(hi)-log(lo))*rand);
    policy=struct('type','expert_noisy','noiseStd',severity,'deterministic',false);
    seed=1000+ep;
    L=sim.runEpisode(policy,'manual',[],seed,cfg,false);
    outcome=2*L.metrics.success-1; % +1 safe landing, -1 otherwise
    K=size(L.graphX,3); idx=1:cfg.rgat.sampleStride:K;
    for k=idx
        Xs{end+1}=L.graphX(:,:,k); %#ok<AGROW>
        % Signed discounted terminal outcome: a potential target, not the manual baseline reward.
        ys(end+1)=outcome*cfg.reward.pbrs.gamma^(K-k); %#ok<AGROW>
        meta(end+1,:)=[ep,k,L.metrics.success,severity]; %#ok<AGROW>
    end
    if isempty(template)
        env=sim.resetState(seed,cfg); c=sim.getCurrent(env,cfg); template=c.graph;
    end
    S.success(ep)=L.metrics.success; S.noise(ep)=severity; S.samples(ep)=numel(idx);
    if cfg.viz.training, update(mon,S,ep); drawnow limitrate; end
    fprintf('  ep %3d/%3d | success=%d | noise=%.2f | samples=%d\n',ep,nEp,L.metrics.success,severity,numel(idx));
end
n=numel(Xs); X=zeros(cfg.ontology.inDim,cfg.ontology.nNodes,n);
for i=1:n, X(:,:,i)=Xs{i}; end
D=struct('X',X,'y',ys,'meta',meta,'graph',template);
fprintf('Dataset: %d samples, positive %.1f%% | successful episodes %d/%d\n', ...
    n,100*mean(ys>0),sum(S.success),nEp);
end
