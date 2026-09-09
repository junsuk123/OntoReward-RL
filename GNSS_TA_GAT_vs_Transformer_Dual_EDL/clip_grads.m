function grads=clip_grads(grads,thr)
for i=1:numel(grads)
    if isempty(grads{i}), continue; end
    g=grads{i};
    n=sqrt(sum(g.^2,'all')+1e-12);
    scale=min(1,thr/n);
    grads{i}=g*scale;
end
end
