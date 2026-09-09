function kl = dirichlet_kl_uniform(alpha)
% KL(Dir(alpha) || Dir(1)), sample-wise.
% Built-in gammaln/psi do not accept dlarray, so the local helpers below
% implement both with plain differentiable ops (valid for alpha >= 1).
K=size(alpha,1);
S=sum(alpha,1);

term0=gammaln_dl(S)-sum(gammaln_dl(alpha),1)-gammaln_dl(single(K));
term1=sum((alpha-1).*(psi_dl(alpha)-psi_dl(S)),1);
kl=term0+term1;
end

function y = gammaln_dl(x)
% Lanczos approximation (g=7), accurate to ~1e-13 relative for x > 0.
g=[676.5203681218851,-1259.1392167224028,771.32342877765313, ...
   -176.61502916214059,12.507343278686905,-0.13857109526572012, ...
   9.9843695780195716e-6,1.5056327351493116e-7];
x=x-1;
a=0.99999999999980993;
for i=1:numel(g)
    a=a+g(i)./(x+i);
end
t=x+7.5;
y=0.5*log(2*pi)+(x+0.5).*log(t)-t+log(a);
end

function y = psi_dl(x)
% Digamma: recurrence psi(x)=psi(x+1)-1/x up to x>=7, then asymptotic series.
s=0;
for k=1:6
    s=s-1./x;
    x=x+1;
end
iv=1./x; iv2=iv.^2;
y=s+log(x)-0.5*iv-iv2.*(1/12-iv2.*(1/120-iv2/252));
end
