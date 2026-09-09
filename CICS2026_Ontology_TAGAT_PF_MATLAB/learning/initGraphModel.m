function model = initGraphModel(pack,G,mode,cfg)
%INITGRAPHMODEL Initialize custom relation-aware graph attention parameters.

F = size(pack.X,1);
D = cfg.learning.hiddenDim;
R = G.numRelations;
N = G.numNodes;

scale = 0.18;

p.Wcommon = dlarray(single(scale*randn(D,F)));
p.Wrel = dlarray(single(scale*randn(D,F,R)));
p.nodeEmb = dlarray(single(scale*randn(D,N)));
p.relEmb = dlarray(single(scale*randn(D,R)));

p.aSrc = dlarray(single(scale*randn(D,1)));
p.aDst = dlarray(single(scale*randn(D,1)));
p.aRel = dlarray(single(scale*randn(D,1)));
p.Wtemp = dlarray(single(scale*randn(D,F)));
p.aTemp = dlarray(single(scale*randn(D,1)));
p.gammaRaw = dlarray(single(0));

p.Wout = dlarray(single(scale*randn(4,4*D)));
p.bout = dlarray(single(zeros(4,1)));

model.params = p;
model.mode = string(mode);
model.window = 1;
if upper(string(mode))=="TAGAT"
    model.window = cfg.learning.window;
end
model.hiddenDim = D;
model.numRelations = R;
end
