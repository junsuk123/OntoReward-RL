function [p,u] = predict_probabilities(model,X,cfg)
%PREDICT_PROBABILITIES Batched inference returning fault probability and uncertainty.
N=size(X,3); bs=max(1,cfg.train.miniBatchSize*2);
p=zeros(1,N,'single'); u=nan(1,N,'single');
for start=1:bs:N
    idx=start:min(start+bs-1,N);
    Xb=permute(X(:,:,idx),[1 3 2]); dlX=make_dl_batch(Xb);
    raw=predict(model.net,dlX); raw=gather(extractdata(raw));
    [pb,ub]=edl_decode(single(raw),model.lossMode);
    p(idx)=single(pb); u(idx)=single(ub);
end
end
