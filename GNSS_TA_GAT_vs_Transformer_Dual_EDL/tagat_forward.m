function [p1,u1,p2,u2,pFused,uFused,a1,a2] = tagat_forward(params,Stats,A,cfg)
% Stats [S x N x B], A [N x N x B]
D=cfg.tagat.hiddenDim;
N=size(Stats,2); B=size(Stats,3);

k=1;
W0=params{k}; b0=params{k+1}; k=k+2;
H=linear3d(W0,Stats,b0);
H=relu(H);

for l=1:cfg.tagat.numLayers
    W=params{k}; as=params{k+1}; ad=params{k+2}; k=k+3;
    Hnew=gat_layer(H,A,W,as,ad,cfg);
    H=relu(layernorm_simple(H+Hnew));
end

latent=mean(H,2);
latent=reshape(latent,D,B);

Wh1=params{k}; bh1=params{k+1}; Wh2=params{k+2}; bh2=params{k+3};

e1=softplus_stable(Wh1*latent+bh1);
e2=softplus_stable(Wh2*latent+bh2);
a1=e1+1; a2=e2+1;

s1=sum(a1,1); s2=sum(a2,1);
p1=a1(2,:)./s1; p2=a2(2,:)./s2;
u1=2./s1; u2=2./s2;

af=(1-u1).*(a1-1)+(1-u2).*(a2-1)+1;
sf=sum(af,1);
pFused=af(2,:)./sf;
uFused=2./sf;
end

function O=gat_layer(H,A,W,as,ad,cfg)
[D,N,B]=size(H);
out=cell(1,B);

for b=1:B
    z=W*H(:,:,b);               % [D x N]
    si=as.'*z;                  % [1 x N]
    dj=ad.'*z;                  % [1 x N]
    e=si.'+dj;                  % [N x N]
    e=leakyrelu_simple(e,cfg.tagat.leakySlope);

    Ab=single(A(:,:,b));
    prior=cfg.graph.priorBias*log(Ab+cfg.graph.edgeFloor);
    mask=single(Ab>cfg.graph.edgeFloor);
    e=e+dlarray(prior)+dlarray((1-mask)*(-1e4));

    att=softmax_dim(e,2);            % each node attends over neighbors
    out{b}=z*att.';              % [D x N]
end

O=cat(3,out{:});
end

function y=leakyrelu_simple(x,a)
y=max(x,0)+a*min(x,0);
end

function Y=linear3d(W,X,b)
[~,T,B]=size(X);
Y=W*reshape(X,size(X,1),T*B)+b;
Y=reshape(Y,size(W,1),T,B);
end

function z=layernorm_simple(x)
mu=mean(x,1);
v=mean((x-mu).^2,1);
z=(x-mu)./sqrt(v+1e-5);
end

function y=softmax_dim(x,dim)
% dlarray/softmax normalizes only over the 'C' label, so do it explicitly.
x=x-max(x,[],dim);
e=exp(x);
y=e./sum(e,dim);
end
