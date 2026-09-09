function [mu,stdv] = actorForward(A,obs)
if ~isa(obs,'dlarray'), obs=dlarray(obs); end
h1=tanh(A.W1*obs+A.b1);
h2=tanh(A.W2*h1+A.b2);
mu=1.5*tanh(A.Wm*h2+A.bm);
stdv=exp(A.logStd);
if size(obs,2)>1, stdv=repmat(stdv,1,size(obs,2)); end
end
