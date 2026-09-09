function validateOntologyGraph(G)
%VALIDATEONTOLOGYGRAPH Lightweight structural validation in MATLAB.
% Full OWL consistency checking should be done offline in Protégé/HermiT.

assert(numel(G.src)==numel(G.dst) && numel(G.dst)==numel(G.rel));
assert(all(G.src>=1 & G.src<=G.numNodes));
assert(all(G.dst>=1 & G.dst<=G.numNodes));
assert(all(G.rel>=1 & G.rel<=G.numRelations));
assert(numel(unique(G.nodeNames))==numel(G.nodeNames),"Duplicate ontology node names.");
assert(all(ismember(G.outputNodes,1:G.numNodes)));

for n=G.outputNodes
    assert(any(G.dst==n),"Output ontology node %d has no incoming relation.",n);
end
end
