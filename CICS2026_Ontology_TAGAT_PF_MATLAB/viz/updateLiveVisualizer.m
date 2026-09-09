function H = updateLiveVisualizer(H,data,testIdx,kk,est,particles,weights,ESS,Nlog,rlog,lamlog,name,cfg) %#ok<INUSD>
%UPDATELIVEVISUALIZER Refresh live diagnostics.

ids=1:kk;
set(H.estLine,"XData",est(ids,1),"YData",est(ids,2));

[~,ord]=sort(weights,"descend");
ord=ord(1:min(cfg.visual.maxParticleDraw,numel(ord)));
set(H.particlePts,"XData",particles(ord,1),"YData",particles(ord,2));

gt=data.gt(testIdx(ids),1:2);
err=sqrt(sum((est(ids,1:2)-gt).^2,2));
tt=data.time(testIdx(ids))-data.time(testIdx(1));
set(H.errLine,"XData",tt,"YData",err);

yyaxis(H.axPF,"left"); set(H.essLine,"XData",tt,"YData",ESS(ids));
yyaxis(H.axPF,"right"); set(H.nLine,"XData",tt,"YData",Nlog(ids));

H.relBars.YData=rlog(kk,:);
drawnow limitrate;
end
