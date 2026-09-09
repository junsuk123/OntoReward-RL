function saveAttentionSummary(pred,G,filePath,idx)
%SAVEATTENTIONSUMMARY Mean edge attention over selected samples.

if isempty(pred.edgeAttention)
    return;
end

A = pred.edgeAttention(:,idx);
meanA = mean(A,2,"omitnan");

% Force column orientation: indexing a scalar string (the untyped GAT variant
% has a single relation name) with a column index returns a column, while
% indexing a row vector returns a row.
srcName = reshape(G.nodeNames(G.src),[],1);
dstName = reshape(G.nodeNames(G.dst),[],1);
relName = reshape(G.relationNames(G.rel),[],1);

T = table(srcName,dstName,relName,meanA, ...
    'VariableNames',["Source","Destination","Relation","MeanAttention"]);
T = sortrows(T,"MeanAttention","descend");
writetable(T,filePath);
end
