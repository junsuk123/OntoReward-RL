function v = criticForward(C,obs)
if ~isa(obs,'dlarray'), obs=dlarray(obs); end
h1=tanh(C.W1*obs+C.b1);
h2=tanh(C.W2*h1+C.b2);
v=C.Wv*h2+C.bv;
end
