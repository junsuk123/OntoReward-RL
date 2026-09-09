function G = buildOntologyGraph(s,cfg)
%BUILDONTOLOGYGRAPH Fixed semantic schema + state-dependent node features.
% Nodes retain domain meaning; R-GAT learns context-dependent relation weights.
N=cfg.ontology.nNodes;
vals=zeros(1,N);
vals(1)=min(1,s.positionError/2.5);
vals(2)=min(1,abs(s.verticalSpeed)/1.5);
vals(3)=min(1,s.tilt/deg2rad(45));
vals(4)=min(1,s.angularRate/deg2rad(180));
vals(5)=s.windRisk;
vals(6)=s.markerQuality;
vals(7)=s.visualStability;
vals(8)=s.alignment;
vals(9)=s.attitudeStability;
vals(10)=s.touchdownSafety;
vals(11)=0; % SafeLanding goal node: no future-label leakage.
riskNodes=[1 2 3 4 5];
X=zeros(cfg.ontology.inDim,N);
for i=1:N
    riskFlag=double(ismember(i,riskNodes));
    X(1:4,i)=[vals(i);1-vals(i);riskFlag;1];
    X(4+i,i)=1; % node identity one-hot
end
% relation ids: 1 degrades, 2 supports, 3 contributes, 4 self
src=[1 5 4 3 6 7 8 9 2 5 10 7 8 9 5];
dst=[8 9 9 9 7 8 10 10 10 10 11 11 11 11 11];
rel=[1 1 1 1 2 2 2 2 1 1 3 3 3 3 1];
for i=1:N
    src(end+1)=i; dst(end+1)=i; rel(end+1)=4; %#ok<AGROW>
end
G=struct('X',X,'src',src,'dst',dst,'rel',rel,'goalNode',11, ...
    'nodeNames',{cfg.ontology.nodeNames},'relationNames',{cfg.ontology.relationNames});
end
