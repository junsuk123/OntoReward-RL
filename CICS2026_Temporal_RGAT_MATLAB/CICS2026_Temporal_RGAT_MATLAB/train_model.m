function [model,history] = train_model(model,X,Yh,Ys,cfg,val)
%TRAIN_MODEL Custom Adam training loop with validation-based model selection.
%
% VAL is an optional struct with fields X, Yh, Ys. When supplied, the epoch with
% the best validation AUROC is kept instead of the last epoch. Without it a run
% simply reports its final epoch, which on this dataset means reporting a model
% that has memorised the training windows: overlapping stride-1 windows let the
% plain Transformer reach a training loss of 0.01 while sitting at chance on
% held-out data. Selecting on AUROC rather than validation loss keeps the choice
% independent of the EDL regulariser's annealing schedule, which is still ramping
% while the model is being selected.
if nargin<6, val=[]; end
net=model.net; N=size(X,3); bs=cfg.train.miniBatchSize;
avg=[]; avgSq=[]; iter=0;
E=cfg.train.epochs;
histEpoch=zeros(E,1); histLoss=zeros(E,1); histValAuroc=nan(E,1); histValLoss=nan(E,1);
useVal=~isempty(val) && ~isempty(val.Yh);
bestScore=-Inf; bestNet=net; bestEpoch=0; sinceBest=0;
patience=Inf;
if isfield(cfg.train,'patience'), patience=cfg.train.patience; end

fprintf('\nTraining %-34s | %d windows | %d epochs\n',model.name,N,E);
for epoch=1:E
    order=randperm(N); epochLoss=0; nBatch=0;
    for start=1:bs:N
        iter=iter+1; idx=order(start:min(start+bs-1,N));
        Xb=permute(X(:,:,idx),[1 3 2]); % C x B x T
        dlX=make_dl_batch(Xb);
        yh=single(Yh(idx)); ys=single(Ys(idx));
        [loss,grads,state]=dlfeval(@model_gradients,net,dlX,yh,ys,model.lossMode,cfg,epoch);
        net.State=state;
        grads=dlupdate(@(g)clip_gradient(g,cfg.train.gradientClip),grads);
        [net,avg,avgSq]=adamupdate(net,grads,avg,avgSq,iter,cfg.train.learnRate, ...
            cfg.train.gradDecay,cfg.train.sqGradDecay);
        epochLoss=epochLoss+double(gather(extractdata(loss))); nBatch=nBatch+1;
    end
    histEpoch(epoch)=epoch; histLoss(epoch)=epochLoss/max(1,nBatch);

    if useVal
        probe=model; probe.net=net;
        [pv,~]=predict_probabilities(probe,val.X,cfg);
        Mv=compute_metrics(val.Yh,pv,cfg.threshold);
        % Report val F1 at its own best threshold. Printed at a fixed 0.5 this
        % column just shows the all-positive degenerate value whenever the split
        % shifts the class prior, which hides real progress in the ranking.
        [~,vf1best]=select_threshold(val.Yh,pv,cfg.threshold);
        histValAuroc(epoch)=Mv.AUROC; histValLoss(epoch)=Mv.NLL;
        fprintf('  epoch %3d/%3d | loss %.5f | val AUROC %.4f | val F1* %.4f\n', ...
            epoch,E,histLoss(epoch),Mv.AUROC,vf1best);
        if Mv.AUROC>bestScore
            bestScore=Mv.AUROC; bestNet=net; bestEpoch=epoch; sinceBest=0;
        else
            sinceBest=sinceBest+1;
            if sinceBest>=patience
                fprintf('  early stop at epoch %d (best epoch %d, val AUROC %.4f)\n', ...
                    epoch,bestEpoch,bestScore);
                histEpoch=histEpoch(1:epoch); histLoss=histLoss(1:epoch);
                histValAuroc=histValAuroc(1:epoch); histValLoss=histValLoss(1:epoch);
                break
            end
        end
    else
        fprintf('  epoch %3d/%3d | loss %.5f\n',epoch,E,histLoss(epoch));
    end
end

if useVal
    net=bestNet;
    probe=model; probe.net=net;
    pv=predict_probabilities(probe,val.X,cfg);
    [model.threshold,vf1]=select_threshold(val.Yh,pv,cfg.threshold);
    fprintf('  selected epoch %d | val AUROC %.4f | threshold %.3f (val F1 %.4f)\n', ...
        bestEpoch,bestScore,model.threshold,vf1);
else
    model.threshold=cfg.threshold;
end
model.net=net;
history=table(histEpoch,histLoss,histValAuroc,histValLoss, ...
    'VariableNames',{'Epoch','Loss','ValAUROC','ValNLL'});
end

function g=clip_gradient(g,thr)
if isempty(g), return; end
g=max(min(g,thr),-thr);
end
