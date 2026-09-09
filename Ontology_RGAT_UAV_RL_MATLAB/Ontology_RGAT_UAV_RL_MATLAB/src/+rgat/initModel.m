function P = initModel(cfg)
%INITMODEL Initialize a two-layer relation-aware graph attention potential model.
rng(cfg.seed+101,'twister');
din=cfg.ontology.inDim; dh=cfg.rgat.hiddenDim; R=cfg.ontology.nRelations; dr=cfg.rgat.relDim;
scale=0.12;
P.W1=dlarray(scale*randn(dh,din,R));
P.a1=dlarray(scale*randn(1,2*dh+dr,R));
P.E1=dlarray(scale*randn(dr,R));
P.W2=dlarray(scale*randn(dh,dh,R));
P.a2=dlarray(scale*randn(1,2*dh+dr,R));
P.E2=dlarray(scale*randn(dr,R));
P.wOut=dlarray(scale*randn(1,dh));
P.bOut=dlarray(0);
end
