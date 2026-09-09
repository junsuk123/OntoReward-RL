function y=psi_dl(x)
%PSI_DL Digamma that works for numeric, dlarray and gpuArray inputs.
% MATLAB's builtin PSI rejects dlarray. Shifted asymptotic series, autodiff-safe.
% Valid for x>0; accuracy is ~1e-12 for x>=1 (all Dirichlet alphas).
n=8; s=0;
for k=0:n-1
    s=s+1./(x+k);
end
z=x+n;
iz=1./z;
y=log(z)-iz/2-iz.^2/12+iz.^4/120-iz.^6/252+iz.^8/240;
y=y-s;
end
