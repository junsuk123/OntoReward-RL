function [Stats,Aall] = build_temporal_graph_inputs(X,Aont,cfg)
% X     [N x W x B]
% Stats [S x N x B]
% Aall  [N x N x B]

[N,W,B] = size(X);
S = cfg.tagat.nodeStatDim;
Stats = zeros(S,N,B,'single');
Aall = zeros(N,N,B,'single');

tt = single(1:W);
tt = tt - mean(tt);
den = sum(tt.^2) + eps('single');

for b=1:B
    x = single(X(:,:,b)); % [N x W]

    % Node temporal statistics
    m = mean(x,2);
    sd = std(x,0,2);
    last = x(:,end);
    delta = x(:,end)-x(:,1);
    slope = (x*tt(:))/den;

    Stats(:,:,b) = [m.'; sd.'; last.'; delta.'; slope.'];

    % A_cov
    C = corrcoef(double(x.'));
    C(~isfinite(C)) = 0;
    Acov = single(abs(C));
    Acov(1:N+1:end) = 1;

    % A_dyn: nonlinear similarity of derivative trajectories
    dx = diff(x,1,2);
    D = zeros(N,N,'single');
    for i=1:N
        for j=1:N
            d = dx(i,:) - dx(j,:);
            D(i,j) = mean(d.^2);
        end
    end
    nz = D(D>0 & isfinite(D));
    if isempty(nz)
        sig2 = single(1);
    else
        sig2 = median(nz);
        if sig2 < 1e-6, sig2 = single(1); end
    end
    Adyn = exp(-D/(2*sig2));
    Adyn(1:N+1:end)=1;

    A = cfg.graph.alphaOnt*Aont + ...
        cfg.graph.alphaCov*Acov + ...
        cfg.graph.alphaDyn*Adyn;

    A = max(A,0);
    A = A ./ max(max(A(:)),eps('single'));
    A(1:N+1:end)=1;
    Aall(:,:,b)=A;
end
end
