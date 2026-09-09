function [loss,gradients,state,parts] = model_gradients(net,X,Yhard,Ysoft,lossMode,cfg,epoch)
%MODEL_GRADIENTS Custom loss for softmax/EDL/Dual-EDL models.
%
% EDL heads follow Sensoy et al. 2018:
%   evidence e = softplus(raw), alpha = e + 1, S = sum(alpha)
%   data term  = expected cross-entropy under Dir(alpha)
%                  sum_j T_j * (psi(S) - psi(alpha_j))
%   regulariser= KL[ Dir(alpha_tilde) || Dir(1) ] with the MISLEADING-evidence
%                alpha_tilde = T + (1-T).*alpha, annealed by lambda_t.
%
% The mask matters. Applying the KL to the full alpha penalises evidence for
% the CORRECT class too, so CE+KL is minimised at zero evidence and every head
% collapses to alpha->1, p->0.5, u->1. That is a silent failure: the loss
% curve still looks like it is converging while the model has learnt nothing.
[raw,state]=forward(net,X);
raw=stripdims(raw); Yhard=single(Yhard); Ysoft=single(Ysoft);
Th=[1-Yhard;Yhard]; Ts=[1-Ysoft;Ysoft];
mode=string(lossMode); parts=struct('hard',0,'weak',0,'klHard',0,'klWeak',0);
lam=min(1,epoch/cfg.edl.annealEpochs);
switch mode
    case "softmax_hard"
        P=softmax_cols(raw); loss=cross_entropy_prob(P,Th);
    case "edl_hard"
        a=softplus_stable(raw)+1;
        ce=edl_expected_ce(a,Th); kl=dirichlet_kl_uniform(edl_mask(a,Th));
        loss=ce+lam*cfg.edl.klHardWeight*kl; parts.hard=ce; parts.klHard=kl;
    case "edl_soft"
        a=softplus_stable(raw)+1;
        ce=edl_expected_ce(a,Ts); kl=dirichlet_kl_uniform(edl_mask(a,Ts));
        loss=ce+lam*cfg.edl.klWeakWeight*kl; parts.weak=ce; parts.klWeak=kl;
    case "dual_edl"
        a1=softplus_stable(raw(1:2,:))+1; a2=softplus_stable(raw(3:4,:))+1;
        ce1=edl_expected_ce(a1,Th); kl1=dirichlet_kl_uniform(edl_mask(a1,Th));
        ce2=edl_expected_ce(a2,Ts); kl2=dirichlet_kl_uniform(edl_mask(a2,Ts));
        Lh=ce1+lam*cfg.edl.klHardWeight*kl1;
        Lw=ce2+lam*cfg.edl.klWeakWeight*kl2;
        loss=Lh+cfg.edl.betaWeak*Lw;
        parts.hard=Lh; parts.weak=Lw; parts.klHard=kl1; parts.klWeak=kl2;
    otherwise
        error('Unknown loss mode %s',mode);
end
gradients=dlgradient(loss,net.Learnables);
end

function at=edl_mask(alpha,T)
%EDL_MASK Misleading-evidence Dirichlet: keep only evidence on wrong classes.
% For a one-hot T this removes the correct class' evidence entirely; for a soft
% T it scales it down in proportion, which is the natural soft generalisation.
at=T+(1-T).*alpha;
end

function L=edl_expected_ce(alpha,T)
%EDL_EXPECTED_CE E_{p~Dir(alpha)}[ CE(p,T) ] = sum_j T_j (psi(S)-psi(alpha_j)).
S=sum(alpha,1);
L=mean(sum(T.*(psi_dl(S)-psi_dl(alpha)),1),'all');
end

function P=softmax_cols(Z)
Z=Z-max(Z,[],1); E=exp(Z); P=E./(sum(E,1)+1e-8);
end
function L=cross_entropy_prob(P,T)
L=-mean(sum(T.*log(P+single(1e-7)),1),'all');
end
