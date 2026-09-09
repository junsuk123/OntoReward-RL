function A = build_ontology_adjacency(features)
% Feature-level engineering prior derived from the user's ontology groups.
% This is intentionally editable and should be ablated in the paper.

N = numel(features);
A = eye(N,'single');

idx = containers.Map(features,1:N);
edge = @(a,b,w) local_edge(a,b,w);

% Semantic groups
addClique({'hDOP','vDOP','low_elev_ratio'},0.9);       % Geometry
addClique({'CN0_mean','CN0_std','CN0_gap'},0.9);       % Signal quality
addClique({'hAcc','vAcc'},0.8);                         % Position accuracy

% Cross-group relations
local_edge('hDOP','hAcc',1.0);
local_edge('vDOP','vAcc',1.0);

local_edge('low_elev_ratio','CN0_mean',0.8);
local_edge('low_elev_ratio','CN0_gap',0.8);
local_edge('low_elev_ratio','hDOP',0.8);
local_edge('low_elev_ratio','vDOP',0.8);

local_edge('CN0_mean','PR_RMS',0.9);
local_edge('CN0_std','PR_RMS',0.7);
local_edge('CN0_gap','PR_RMS',0.9);

local_edge('PR_RMS','hAcc',0.8);
local_edge('PR_RMS','vAcc',0.8);
local_edge('PR_RMS','Fault_SVID_count',1.0);

local_edge('CN0_mean','Fault_SVID_count',0.8);
local_edge('CN0_std','Fault_SVID_count',0.7);
local_edge('CN0_gap','Fault_SVID_count',0.8);

local_edge('numSV','hDOP',0.8);
local_edge('numSV','vDOP',0.8);
local_edge('numSV','CN0_mean',0.6);
local_edge('numSV','Fault_SVID_count',0.8);

local_edge('gSpeed','PR_RMS',0.5);
local_edge('gSpeed','hAcc',0.4);
local_edge('gSpeed','vAcc',0.4);

A = max(A,A.');
A = A ./ max(A(:));

    function addClique(names,w)
        for ii=1:numel(names)
            for jj=ii+1:numel(names)
                local_edge(names{ii},names{jj},w);
            end
        end
    end

    function local_edge(a,b,w)
        if isKey(idx,a) && isKey(idx,b)
            ia=idx(a); ib=idx(b);
            A(ia,ib)=max(A(ia,ib),single(w));
            A(ib,ia)=max(A(ib,ia),single(w));
        end
    end
end
