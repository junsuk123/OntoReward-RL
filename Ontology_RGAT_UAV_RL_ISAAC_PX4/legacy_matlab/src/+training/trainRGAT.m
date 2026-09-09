function [P,H] = trainRGAT(D,cfg)
%TRAINRGAT Supervise graph potential with future safe-landing outcome.
%   Same optimizer trajectory as the original -- same initialization, same
%   split, same shuffle order, same Adam steps -- but each step is one batched
%   forward pass instead of cfg.rgat.batchSize of them, and the validation pass
%   is one call instead of one per sample.
%
%   GPU placement is decided by rgat.device from the configured batch size and
%   declines by default, because at cfg.rgat.batchSize=32 the GPU is slower than
%   the vectorized CPU path. See rgat.device for the measurements.
P=rgat.initModel(cfg); opt=struct();
N=size(D.X,3); order=randperm(N); nTrain=max(1,round(0.8*N));
tr=order(1:nTrain); va=order(nTrain+1:end); if isempty(va), va=tr; end
H.trainLoss=zeros(cfg.rgat.epochs,1); H.valLoss=zeros(cfg.rgat.epochs,1);
if cfg.viz.training, mon=viz.RGATMonitor(cfg); else, mon=[]; end

[useGPU,why]=rgat.device(cfg,cfg.rgat.batchSize);
precision='single';
if isfield(cfg,'gpu') && isfield(cfg.gpu,'precision'), precision=cfg.gpu.precision; end
X=D.X; y=D.y;
if useGPU
    cpuModel=P;
    try
        P=rgat.toGPU(P,precision);
        X=gpuArray(cast(X,precision)); y=gpuArray(cast(y,precision));
        fprintf('R-GAT training on the GPU (%s): %s.\n',precision,why);
    catch err
        warning('training:rgatGPU', ...
            'GPU setup failed (%s); continuing on the CPU.',err.message);
        useGPU=false; P=cpuModel; X=D.X; y=D.y;
    end
end
if ~useGPU
    fprintf('R-GAT training on the CPU (vectorized): %s.\n',why);
end
step=0;
for ep=1:cfg.rgat.epochs
    tr=tr(randperm(numel(tr))); losses=zeros(1,ceil(numel(tr)/cfg.rgat.batchSize)); nb=0;
    for s=1:cfg.rgat.batchSize:numel(tr)
        id=tr(s:min(s+cfg.rgat.batchSize-1,numel(tr))); step=step+1; nb=nb+1;
        [loss,G]=dlfeval(@training.rgatGradients,P,X(:,:,id),y(id),D.graph);
        [P,opt]=training.adamStep(P,G,opt,cfg.rgat.lr,step,cfg.ppo.gradClip);
        losses(nb)=double(gather(extractdata(loss)));
    end
    H.trainLoss(ep)=mean(losses(1:nb));
    pv=double(gather(extractdata(rgat.forward(P,X(:,:,va),D.graph))));
    H.valLoss(ep)=mean((pv-double(D.y(va))).^2);
    if cfg.viz.training, update(mon,H,ep,D.y(va),pv); drawnow limitrate; end
    fprintf('R-GAT epoch %3d/%3d | train %.4f | val %.4f\n',ep,cfg.rgat.epochs,H.trainLoss(ep),H.valLoss(ep));
end
% Always hand back a CPU double model: everything downstream (the PBRS reward
% inside the real-time loop, rgat.explain, the saved MAT file) is CPU numeric,
% and a gpuArray in a MAT file cannot be loaded on a machine without the device.
P=rgat.toCPU(P);
end
