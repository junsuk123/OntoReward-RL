function Pn = unwrap(P)
%UNWRAP Plain numeric copy of a parameter struct.
%   extractdata only unwraps the container; it is the traced arithmetic that is
%   expensive, so this is cheap and lets inference skip dlarray entirely.
Pn=P;
names=fieldnames(P);
for k=1:numel(names)
    v=P.(names{k});
    if isa(v,'dlarray'), v=extractdata(v); end
    if isa(v,'gpuArray'), v=gather(v); end
    Pn.(names{k})=double(v);
end
end
