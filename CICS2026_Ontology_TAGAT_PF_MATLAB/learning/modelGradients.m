function [loss,gr] = modelGradients(p,batchX,batchY,G,temporalMode,weightDecay)
%MODELGRADIENTS Mini-batch reliability regression loss.

B = size(batchX,4);
loss = dlarray(single(0));

for b=1:B
    y = graphNetForward(p,batchX(:,:,:,b),G,temporalMode);
    target = dlarray(single(batchY(:,b)));
    loss = loss + mean((y-target).^2,"all");
end
loss = loss / B;

reg = sum(p.Wcommon.^2,"all") + sum(p.Wrel.^2,"all") + ...
      sum(p.Wout.^2,"all") + sum(p.Wtemp.^2,"all");
loss = loss + weightDecay*reg;

[gr.Wcommon,gr.Wrel,gr.nodeEmb,gr.relEmb, ...
 gr.aSrc,gr.aDst,gr.aRel,gr.Wtemp,gr.aTemp,gr.gammaRaw, ...
 gr.Wout,gr.bout] = dlgradient(loss, ...
 p.Wcommon,p.Wrel,p.nodeEmb,p.relEmb, ...
 p.aSrc,p.aDst,p.aRel,p.Wtemp,p.aTemp,p.gammaRaw, ...
 p.Wout,p.bout);
end
