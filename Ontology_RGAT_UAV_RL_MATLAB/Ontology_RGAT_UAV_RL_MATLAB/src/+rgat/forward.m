function phi = forward(P,X,graph)
%FORWARD Predict bounded potential Phi(G) in [-1,1].
if ~isa(X,'dlarray'), X=dlarray(X); end
H1=tanh(rgat.relationLayer(X,P.W1,P.a1,P.E1,graph));
H2=tanh(rgat.relationLayer(H1,P.W2,P.a2,P.E2,graph)+H1);
phi=tanh(P.wOut*H2(:,graph.goalNode)+P.bOut);
end
