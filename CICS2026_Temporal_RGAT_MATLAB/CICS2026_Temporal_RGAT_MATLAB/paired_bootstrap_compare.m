function S = paired_bootstrap_compare(y,pBase,pNew,thr,nBoot,seed)
%PAIRED_BOOTSTRAP_COMPARE Paired bootstrap CI for F1 difference New-Base.
% THR may be a scalar or [thrBase thrNew]: each model is compared at the
% threshold that was selected for it on validation, otherwise the difference
% would confound ranking quality with an arbitrary shared operating point.
if isscalar(thr), thr=[thr thr]; end
rng(seed,'twister'); y=y(:); pBase=pBase(:); pNew=pNew(:); N=numel(y);
d=zeros(nBoot,1);
for b=1:nBoot
    idx=randi(N,N,1);
    mb=compute_metrics(y(idx),pBase(idx),thr(1)); mn=compute_metrics(y(idx),pNew(idx),thr(2));
    d(b)=mn.F1-mb.F1;
end
q=sort(d); lo=q(max(1,round(0.025*nBoot))); hi=q(min(nBoot,round(0.975*nBoot)));
pTwo=2*min(mean(d<=0),mean(d>=0)); pTwo=min(1,pTwo);
S=struct('MeanDeltaF1',mean(d),'CI95Low',lo,'CI95High',hi,'Pbootstrap',pTwo);
end
