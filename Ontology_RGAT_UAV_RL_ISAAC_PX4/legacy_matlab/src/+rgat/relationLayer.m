function Z = relationLayer(H,W,a,E,graph)
%RELATIONLAYER Relation-specific transform + destination-normalized attention.
%   Batched replacement for the original per-node, per-edge loop. The arithmetic
%   is identical, including the leaky-ReLU score and the unshifted softmax with
%   its 1e-9 denominator floor, so a model trained or evaluated either way gives
%   the same numbers; matlab/src/+rgat/private is not needed because the
%   equivalence is covered by tests/test_rgat_equivalence.m.
%
%   H may be [din x N] or [din x N x B]. The original called this once per graph
%   and once per node inside that, which cost roughly 26 tiny dlarray products
%   per layer; two potential evaluations per control step then took 28 ms
%   against a 20 ms control period, so the ontology-RGAT arm could not hold
%   50 Hz while the manual baseline could. That was a bias in the comparison,
%   not just a slow path.
T=rgat.topology(graph);
if ndims(H)<3, H=reshape(H,size(H,1),size(H,2),1); end
[din,N,B]=size(H);
dout=size(W,1); R=size(W,3); Ne=T.nEdges;
if N~=T.nNodes, error('rgat:nodes','Feature matrix has %d nodes, graph has %d.',N,T.nNodes); end
Hf=reshape(H,din,N*B);
hw=cell(1,R); an=cell(1,R); bn=cell(1,R); sc=cell(1,R);
for r=1:R
    HWr=W(:,:,r)*Hf;                       % [dout x N*B] all nodes at once
    hw{r}=reshape(HWr,dout,N,B);
    an{r}=a(1,1:dout,r)*HWr;               % source half of the attention score
    bn{r}=a(1,dout+1:2*dout,r)*HWr;        % destination half
    sc{r}=a(1,2*dout+1:end,r)*E(:,r);      % relation embedding term, scalar
end
HWall=cat(2,hw{:});                        % [dout x N*R x B]
Aall=reshape(cat(1,an{:}),R*N,B);
Ball=reshape(cat(1,bn{:}),R*N,B);
scAll=cat(1,sc{:});                        % [R x 1]
raw=Aall(T.idxSrc,:)+Ball(T.idxDst,:)+scAll(T.rel,ones(1,B));
score=0.6*raw+0.4*abs(raw);                % leaky ReLU, slope 0.2
ex=exp(score);
den=T.Msel*ex+1e-9;                        % [N x B] sum over incoming edges
alpha=ex./den(T.dst,:);                    % destination-normalized attention
msgs=HWall(:,T.colIdx,:);                  % source projections, per edge
weighted=msgs.*reshape(alpha,1,Ne,B);
Z=permute(reshape(T.Msel*reshape(permute(weighted,[2 1 3]),Ne,dout*B),N,dout,B),[2 1 3]);
if B==1, Z=reshape(Z,dout,N); end
if size(Z,1)~=dout, error('R-GAT output dimension mismatch.'); end
end
