function G2 = makeGraphVariant(G,mode)
%MAKEGRAPHVARIANT Build fully-connected untyped GAT baseline or return ontology graph.

if upper(string(mode)) ~= "GAT"
    G2 = G;
    return;
end

src=[]; dst=[];
for d=1:G.numNodes
    for s=1:G.numNodes
        src(end+1)=s; dst(end+1)=d; %#ok<AGROW>
    end
end

G2 = G;
G2.src = src(:);
G2.dst = dst(:);
G2.rel = ones(numel(src),1);
G2.relationNames = "untyped";
G2.numRelations = 1;
end
