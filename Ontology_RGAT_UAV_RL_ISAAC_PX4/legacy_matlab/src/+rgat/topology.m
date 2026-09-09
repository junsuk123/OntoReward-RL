function T = topology(graph)
%TOPOLOGY Edge index tables for the batched relation layer.
%   The ontology schema is fixed for a whole run, so everything that depends
%   only on (src,dst,rel) is computed once and cached. Recomputing it per
%   forward pass would give back a good part of what vectorizing won.
%
%   Cached on the topology itself rather than on a handle the callers would
%   have to thread through every reward and evaluation function in the
%   read-only original workspace.
persistent keyed cached
src=double(graph.src(:)'); dst=double(graph.dst(:)'); rel=double(graph.rel(:)');
N=size(graph.X,2); R=max(rel); key=[N R src dst rel];
if ~isempty(keyed) && isequal(keyed,key)
    T=cached; return;
end
Ne=numel(src);
T=struct();
T.src=src; T.dst=dst; T.rel=rel;
T.nNodes=N; T.nRel=R; T.nEdges=Ne; T.goalNode=double(graph.goalNode);
% Row into the [R x N] per-relation node planes, and column into the
% [dout x N*R] stack of relation-projected node features.
T.idxSrc=rel+R*(src-1);
T.idxDst=rel+R*(dst-1);
T.colIdx=src+N*(rel-1);
% Destination incidence. Full, not sparse: it is N-by-Ne and a sparse operand
% would not survive dlarray tracing.
T.Msel=zeros(N,Ne);
T.Msel(sub2ind([N Ne],dst,1:Ne))=1;
keyed=key; cached=T;
end
