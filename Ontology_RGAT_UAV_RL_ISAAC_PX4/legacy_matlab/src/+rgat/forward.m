function phi = forward(P,X,graph)
%FORWARD Predict bounded potential Phi(G) in [-1,1], for one graph or a batch.
%   X is [inDim x nNodes] or [inDim x nNodes x B]; phi is [1 x B]. The batched
%   form is what makes a training step one pass instead of B passes.
if ~isa(X,'dlarray') && (isa(P.W1,'dlarray') || isa(P.W1,'gpuArray'))
    X=cast(X,'like',P.W1);
end
if ~isa(X,'dlarray') && isa(P.W1,'dlarray'), X=dlarray(X); end
H1=tanh(rgat.relationLayer(X,P.W1,P.a1,P.E1,graph));
H2=tanh(rgat.relationLayer(H1,P.W2,P.a2,P.E2,graph)+H1);
T=rgat.topology(graph);
dh=size(H2,1); B=size(X,3);
phi=tanh(P.wOut*reshape(H2(:,T.goalNode,:),dh,B)+P.bOut);
end
