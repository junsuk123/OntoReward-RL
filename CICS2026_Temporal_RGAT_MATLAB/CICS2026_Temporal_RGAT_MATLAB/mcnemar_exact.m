function S=mcnemar_exact(y,pA,pB,thr)
%MCNEMAR_EXACT Exact two-sided McNemar test without Statistics Toolbox.
if isscalar(thr), thr=[thr thr]; end
y=logical(y(:)); a=(pA(:)>=thr(1))==y; b=(pB(:)>=thr(2))==y;
n01=sum(~a & b); n10=sum(a & ~b); n=n01+n10;
if n==0, p=1; else
    k=min(n01,n10); logs=zeros(k+1,1);
    for i=0:k, logs(i+1)=gammaln(n+1)-gammaln(i+1)-gammaln(n-i+1)-n*log(2); end
    m=max(logs); tail=exp(m)*sum(exp(logs-m)); p=min(1,2*tail);
end
S=struct('BaseWrong_NewCorrect',n01,'BaseCorrect_NewWrong',n10,'Pexact',p);
end
