function [p1,u1,p2,u2,pFused,uFused,a1,a2] = baseline_forward(params,X,cfg)
% X [C x W x B]
D=cfg.transformer.dModel;
H=cfg.transformer.numHeads;
L=cfg.transformer.numLayers;
W=size(X,2); B=size(X,3);

k=1;
Wemb=params{k}; bemb=params{k+1}; k=k+2;

Z = linear3d(Wemb,X,bemb);
pe = positional_encoding(D,W);
Z = Z + dlarray(repmat(pe,1,1,B));

for l=1:L
    Wq=params{k}; Wk=params{k+1}; Wv=params{k+2}; Wo=params{k+3};
    W1=params{k+4}; b1=params{k+5}; W2=params{k+6}; b2=params{k+7};
    k=k+8;

    A = mha_forward(Z,Wq,Wk,Wv,Wo,H);
    Z = layernorm_simple(Z + A);

    F = linear3d(W1,Z,b1);
    F = relu(F);
    F = linear3d(W2,F,b2);
    Z = layernorm_simple(Z + F);
end

latent = mean(Z,2);        % [D x 1 x B]
latent = reshape(latent,D,B);

Wh1=params{k}; bh1=params{k+1}; Wh2=params{k+2}; bh2=params{k+3};

e1=softplus_stable(Wh1*latent + bh1);
e2=softplus_stable(Wh2*latent + bh2);
a1=e1+1; a2=e2+1;

s1=sum(a1,1); s2=sum(a2,1);
p1=a1(2,:)./s1; p2=a2(2,:)./s2;
u1=2./s1; u2=2./s2;

af=(1-u1).*(a1-1)+(1-u2).*(a2-1)+1;
sf=sum(af,1);
pFused=af(2,:)./sf;
uFused=2./sf;
end

function Y=linear3d(W,X,b)
[~,T,B]=size(X);
Y = W*reshape(X,size(X,1),T*B) + b;
Y = reshape(Y,size(W,1),T,B);
end

function O=mha_forward(Z,Wq,Wk,Wv,Wo,H)
[D,T,B]=size(Z);
dh=D/H;
Q=linear3d(Wq,Z,zeros_like(Wq,D,1));
K=linear3d(Wk,Z,zeros_like(Wk,D,1));
V=linear3d(Wv,Z,zeros_like(Wv,D,1));

bout=cell(1,B);
for b=1:B
    hout=cell(1,H);
    for h=1:H
        ix=(h-1)*dh+1:h*dh;
        q=Q(ix,:,b); kk=K(ix,:,b); v=V(ix,:,b);
        score=(q.'*kk)/sqrt(single(dh)); % [T x T]
        att=softmax_dim(score,2);
        hout{h}=v*att.';
    end
    bout{b}=cat(1,hout{:});
end
O=cat(3,bout{:});
O=linear3d(Wo,O,zeros_like(Wo,D,1));
end

function z=layernorm_simple(x)
mu=mean(x,1);
v=mean((x-mu).^2,1);
z=(x-mu)./sqrt(v+1e-5);
end

function pe=positional_encoding(D,T)
pe=zeros(D,T,'single');
pos=single((0:T-1)');
for i=1:2:D
    div=exp(single(-(i-1))*log(single(10000))/single(D));
    pe(i,:)=sin(pos*div).';
    if i+1<=D
        pe(i+1,:)=cos(pos*div).';
    end
end
end

function z=zeros_like(ref,r,c)
z=dlarray(zeros(r,c,'like',extractdata(ref)));
end

function y=softmax_dim(x,dim)
% dlarray/softmax normalizes only over the 'C' label, so do it explicitly.
x=x-max(x,[],dim);
e=exp(x);
y=e./sum(e,dim);
end
