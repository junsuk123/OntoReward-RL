function Eout = explain(P,graph)
%EXPLAIN Numeric second-layer edge attention for interpretability only.
% Attention is learned importance, not causal proof.
fn=fieldnames(P); P2=struct();
for ii=1:numel(fn), P2.(fn{ii})=double(extractdata(P.(fn{ii}))); end
X=graph.X;
H1=tanh(numericLayer(X,P2.W1,P2.a1,P2.E1,graph));
[H2,alpha]=numericLayer(H1,P2.W2,P2.a2,P2.E2,graph); %#ok<ASGLU>
Eout.edgeAlpha=alpha; Eout.src=graph.src; Eout.dst=graph.dst; Eout.rel=graph.rel;
Eout.relationMean=zeros(1,max(graph.rel));
for r=1:max(graph.rel)
    ix=graph.rel==r; Eout.relationMean(r)=mean(alpha(ix));
end
Eout.relationNames=graph.relationNames; Eout.nodeNames=graph.nodeNames;
end

function [Z,alphaAll]=numericLayer(H,W,a,E,graph)
N=size(H,2); Z=zeros(size(W,1),N); alphaAll=zeros(1,numel(graph.src));
for j=1:N
    ee=find(graph.dst==j); scores=zeros(1,numel(ee)); msgs=zeros(size(W,1),numel(ee));
    for kk=1:numel(ee)
        e=ee(kk); i=graph.src(e); r=graph.rel(e);
        hs=W(:,:,r)*H(:,i); hd=W(:,:,r)*H(:,j); raw=a(:,:,r)*[hs;hd;E(:,r)];
        scores(kk)=max(raw,0.2*raw); msgs(:,kk)=hs;
    end
    ex=exp(scores-max(scores)); al=ex/(sum(ex)+1e-12); Z(:,j)=sum(msgs.*al,2); alphaAll(ee)=al;
end
end
