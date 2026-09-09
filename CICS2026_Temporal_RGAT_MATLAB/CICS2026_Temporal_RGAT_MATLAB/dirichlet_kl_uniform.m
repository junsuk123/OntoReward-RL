function L=dirichlet_kl_uniform(alpha)
%DIRICHLET_KL_UNIFORM Mean KL[Dir(alpha)||Dir(1)] over batch.
% Uses autodiff-safe gammaln_dl/psi_dl because the builtin GAMMALN and PSI
% reject dlarray inputs and would break the EDL gradient path.
K=size(alpha,1); S=sum(alpha,1);
term1=gammaln_dl(S)-sum(gammaln_dl(alpha),1)-gammaln_dl(single(K));
term2=sum((alpha-1).*(psi_dl(alpha)-psi_dl(S)),1);
L=mean(term1+term2,'all');
end
