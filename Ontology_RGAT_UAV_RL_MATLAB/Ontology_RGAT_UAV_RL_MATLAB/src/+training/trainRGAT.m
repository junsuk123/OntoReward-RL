function [P,H] = trainRGAT(D,cfg)
%TRAINRGAT Supervise graph potential with future safe-landing outcome.
P=rgat.initModel(cfg); opt=struct();
N=size(D.X,3); order=randperm(N); nTrain=max(1,round(0.8*N));
tr=order(1:nTrain); va=order(nTrain+1:end); if isempty(va), va=tr; end
H.trainLoss=zeros(cfg.rgat.epochs,1); H.valLoss=zeros(cfg.rgat.epochs,1);
if cfg.viz.training, mon=viz.RGATMonitor(cfg); else, mon=[]; end
step=0;
for ep=1:cfg.rgat.epochs
    tr=tr(randperm(numel(tr))); losses=[];
    for s=1:cfg.rgat.batchSize:numel(tr)
        id=tr(s:min(s+cfg.rgat.batchSize-1,numel(tr))); step=step+1;
        [loss,G]=dlfeval(@training.rgatGradients,P,D.X(:,:,id),D.y(id),D.graph);
        [P,opt]=training.adamStep(P,G,opt,cfg.rgat.lr,step,cfg.ppo.gradClip);
        losses(end+1)=double(extractdata(loss)); %#ok<AGROW>
    end
    H.trainLoss(ep)=mean(losses);
    pv=zeros(1,numel(va));
    for i=1:numel(va), pv(i)=double(extractdata(rgat.forward(P,D.X(:,:,va(i)),D.graph))); end
    H.valLoss(ep)=mean((pv-D.y(va)).^2);
    if cfg.viz.training, update(mon,H,ep,D.y(va),pv); drawnow limitrate; end
    fprintf('R-GAT epoch %3d/%3d | train %.4f | val %.4f\n',ep,cfg.rgat.epochs,H.trainLoss(ep),H.valLoss(ep));
end
end
