function [loss,G] = rgatGradients(P,Xbatch,ybatch,graph)
B=size(Xbatch,3); pc=cell(1,B);
for b=1:B
    pc{b}=rgat.forward(P,Xbatch(:,:,b),graph);
end
pred=cat(2,pc{:});
y=dlarray(ybatch);
% MSE plus weak output regularization.
loss=mean((pred-y).^2,'all')+1e-4*mean(pred.^2,'all');
[G.W1,G.a1,G.E1,G.W2,G.a2,G.E2,G.wOut,G.bOut]=dlgradient(loss, ...
    P.W1,P.a1,P.E1,P.W2,P.a2,P.E2,P.wOut,P.bOut);
end
