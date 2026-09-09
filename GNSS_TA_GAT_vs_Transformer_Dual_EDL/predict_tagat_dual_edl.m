function [p,u] = predict_tagat_dual_edl(model,Stats,A,cfg)
B=size(Stats,3);
p=zeros(1,B,'single'); u=p;
bs=cfg.training.batchSize;
for s=1:bs:B
    id=s:min(s+bs-1,B);
    st=dlarray(single(Stats(:,:,id)));
    ab=single(A(:,:,id));
    [~,~,~,~,pf,uf]=tagat_forward(model.params,st,ab,cfg);
    p(id)=gather(extractdata(pf));
    u(id)=gather(extractdata(uf));
end
end
