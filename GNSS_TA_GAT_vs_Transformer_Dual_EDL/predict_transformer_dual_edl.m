function [p,u] = predict_transformer_dual_edl(model,X,cfg)
B=size(X,3);
p=zeros(1,B,'single'); u=p;
bs=cfg.training.batchSize;
for s=1:bs:B
    id=s:min(s+bs-1,B);
    xb=dlarray(single(X(:,:,id)));
    [~,~,~,~,pf,uf] = baseline_forward(model.params,xb,cfg);
    p(id)=gather(extractdata(pf));
    u(id)=gather(extractdata(uf));
end
end
