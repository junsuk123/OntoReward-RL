function [batchX,batchY] = makeBatchWindows(pack,idx,window)
%MAKEBATCHWINDOWS Construct padded sliding windows [F N W B].

F=size(pack.X,1); N=size(pack.X,2); B=numel(idx);
batchX = zeros(F,N,window,B,"single");
batchY = zeros(4,B,"single");

for b=1:B
    t = idx(b);
    ids = max(1,t-window+1):t;
    if numel(ids)<window
        ids = [repmat(ids(1),1,window-numel(ids)),ids];
    end
    batchX(:,:,:,b) = pack.X(:,:,ids);
    batchY(:,b) = pack.Y(:,t);
end
end
