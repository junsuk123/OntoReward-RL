function y=gammaln_dl(x)
%GAMMALN_DL log-gamma that works for numeric, dlarray and gpuArray inputs.
% MATLAB's builtin GAMMALN rejects dlarray, so the EDL Dirichlet KL term cannot
% be differentiated with it. This uses the shifted Stirling series, which is
% built only from log/power/divide and is therefore fully autodiff-traceable.
% Valid for x>0; accuracy is ~1e-12 relative for x>=1 (all Dirichlet alphas).
n=8; s=0;
for k=0:n-1
    s=s+log(x+k);
end
z=x+n;
iz=1./z;
y=(z-0.5).*log(z)-z+0.5*log(2*pi) ...
    +iz/12-iz.^3/360+iz.^5/1260-iz.^7/1680;
y=y-s;
end
