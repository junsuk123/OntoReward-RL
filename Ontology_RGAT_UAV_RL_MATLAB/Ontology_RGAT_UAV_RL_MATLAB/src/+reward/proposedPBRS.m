function [r,parts] = proposedPBRS(cur,next,status,model,cfg)
%PROPOSEDPBRS Ontology-RGAT potential-based reward shaping.
base=reward.sparseTask(status,cfg);
phi0=rgat.predict(model,cur.graph);
if strcmp(status,'running')
    phi1=rgat.predict(model,next.graph);
else
    phi1=0; % absorbing terminal state potential, preserving PBRS form
end
shape=cfg.reward.pbrs.lambda*(cfg.reward.pbrs.gamma*phi1-phi0);
r=base+shape;
parts=struct('base',base,'shape',shape,'phi0',phi0,'phi1',phi1);
end
