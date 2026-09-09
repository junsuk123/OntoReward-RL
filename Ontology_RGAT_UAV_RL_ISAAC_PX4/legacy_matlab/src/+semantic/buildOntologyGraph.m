function G = buildOntologyGraph(s,cfg)
%BUILDONTOLOGYGRAPH Fixed semantic schema + state-dependent node features.
% Nodes retain domain meaning; R-GAT learns context-dependent relation weights.
%
% External version. Two nodes are appended before the goal:
%
%   11 PadMotion       risk. The pad rides a ground vehicle: its own speed and
%                      the velocity still to be matched both make a landing
%                      harder, and both blur the markers the pose depends on.
%   12 BatteryReserve  support. Energy still available after paying for the
%                      descent that remains; 1 means the reserve outlasts any
%                      attempt, 0 means there is nothing left to spend.
%
% SafeLanding moves to 13, so a model trained against the original 11-node
% schema is not loadable here; rgat.forward errors on the dimension rather than
% quietly attending over the wrong nodes.
N=cfg.ontology.nNodes;
if N~=13
    error('semantic:nNodes', ...
        ['The external ontology has 13 nodes (PadMotion and BatteryReserve ' ...
        'before SafeLanding) but cfg.ontology.nNodes is %d. Build cfg with ' ...
        'defaultExternalConfig, not defaultConfig.'],N);
end
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
vals(11)=s.padMotion;
vals(12)=s.batteryReserve;
vals(13)=0; % SafeLanding goal node: no future-label leakage.
riskNodes=[1 2 3 4 5 11];
X=zeros(cfg.ontology.inDim,N);
for i=1:N
    riskFlag=double(ismember(i,riskNodes));
    X(1:4,i)=[vals(i);1-vals(i);riskFlag;1];
    X(4+i,i)=1; % node identity one-hot
end
% relation ids: 1 degrades, 2 supports, 3 contributes, 4 self
%       original schema, with SafeLanding renumbered 11 -> 13
src=[1 5 4 3 6 7 8 9 2 5 10 7 8 9 5];
dst=[8 9 9 9 7 8 10 10 10 10 13 13 13 13 13];
rel=[1 1 1 1 2 2 2 2 1 1  3  3  3  3  1];
%       PadMotion degrades alignment, the pose it is solved from, the touchdown
%       itself, and the goal; BatteryReserve supports the touchdown and
%       contributes to the goal the same way the other enablers do.
src=[src 11 11 11 11 12 12];
dst=[dst  8  7 10 13 10 13];
rel=[rel  1  1  1  1  2  3];
for i=1:N
    src(end+1)=i; dst(end+1)=i; rel(end+1)=4; %#ok<AGROW>
end
G=struct('X',X,'src',src,'dst',dst,'rel',rel,'goalNode',13, ...
    'nodeNames',{cfg.ontology.nodeNames},'relationNames',{cfg.ontology.relationNames});
end
