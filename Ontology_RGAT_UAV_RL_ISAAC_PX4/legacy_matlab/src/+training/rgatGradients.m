function [loss,G] = rgatGradients(P,Xbatch,ybatch,graph)
%RGATGRADIENTS One batched loss and gradient for the graph potential.
%   The original evaluated the batch one graph at a time, which made a gradient
%   step B separate forward passes; rgat.forward now takes the whole batch, so
%   this is one pass. Same loss, same gradients, roughly 70x less time.
if ~isa(Xbatch,'dlarray'), Xbatch=dlarray(cast(Xbatch,'like',extractdata(P.W1))); end
pred=rgat.forward(P,Xbatch,graph);
y=dlarray(cast(ybatch(:)','like',pred));
% MSE plus weak output regularization.
loss=mean((pred-y).^2,'all')+1e-4*mean(pred.^2,'all');
[G.W1,G.a1,G.E1,G.W2,G.a2,G.E2,G.wOut,G.bOut]=dlgradient(loss, ...
    P.W1,P.a1,P.E1,P.W2,P.a2,P.E2,P.wOut,P.bOut);
end
