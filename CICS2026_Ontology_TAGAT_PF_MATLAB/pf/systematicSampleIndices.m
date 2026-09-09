function idx = systematicSampleIndices(w,Nout)
%SYSTEMATICSAMPLEINDICES Systematic resampling to arbitrary output count.

w=w(:); w=w/sum(w);
cdf=cumsum(w);
u0=rand/Nout;
u=u0+(0:Nout-1)'/Nout;
idx=zeros(Nout,1);
j=1;
for i=1:Nout
    while j<numel(cdf) && u(i)>cdf(j)
        j=j+1;
    end
    idx(i)=j;
end
end
