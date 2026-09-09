function P = toCPU(P)
%TOCPU Gather a parameter struct back to CPU double dlarray.
names=fieldnames(P);
for k=1:numel(names)
    v=P.(names{k});
    if isa(v,'dlarray'), v=extractdata(v); end
    if isa(v,'gpuArray'), v=gather(v); end
    P.(names{k})=dlarray(double(v));
end
end
