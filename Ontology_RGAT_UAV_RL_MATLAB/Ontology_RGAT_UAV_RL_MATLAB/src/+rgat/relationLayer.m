function Z = relationLayer(H,W,a,E,graph)
%RELATIONLAYER Relation-specific transform + destination-normalized attention.
N=size(H,2); dout=size(W,1); cols=cell(1,N);
for j=1:N
    ee=find(graph.dst==j);
    scoreCell=cell(1,numel(ee)); msgCell=cell(1,numel(ee));
    for kk=1:numel(ee)
        e=ee(kk); i=graph.src(e); r=graph.rel(e);
        hs=W(:,:,r)*H(:,i);
        hd=W(:,:,r)*H(:,j);
        z=[hs;hd;E(:,r)];
        raw=a(:,:,r)*z;
        score=0.6*raw+0.4*abs(raw); % leaky-ReLU with slope 0.2
        scoreCell{kk}=score; msgCell{kk}=hs;
    end
    scores=cat(2,scoreCell{:}); msgs=cat(2,msgCell{:});
    ex=exp(scores);
    alpha=ex/(sum(ex)+1e-9);
    cols{j}=sum(msgs.*alpha,2);
end
Z=cat(2,cols{:});
if size(Z,1)~=dout, error('R-GAT output dimension mismatch.'); end
end
