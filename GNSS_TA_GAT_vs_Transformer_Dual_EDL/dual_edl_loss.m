function loss = dual_edl_loss(aHard,aSoft,yHardFault,ySoftFault,cfg,epoch)
% aHard/aSoft: [2 x B] Dirichlet parameters
% Labels converted to [normal; fault]

yh=dlarray(single([1-yHardFault; yHardFault]));
ys=dlarray(single([1-ySoftFault; ySoftFault]));

ph=aHard./sum(aHard,1);
ps=aSoft./sum(aSoft,1);

ceH=-mean(sum(yh.*log(ph+1e-7),1));
ceS=-mean(sum(ys.*log(ps+1e-7),1));

if cfg.loss.useKLD
    klH=mean(dirichlet_kl_uniform(aHard));
    klS=mean(dirichlet_kl_uniform(aSoft));
else
    % Robust fallback when psi/gammaln autodiff is unavailable:
    % penalize evidence assigned to non-target classes.
    eH=aHard-1; eS=aSoft-1;
    klH=mean(sum((1-yh).*eH,1));
    klS=mean(sum((1-ys).*eS,1));
end

lambda=min(1,epoch/max(cfg.loss.annealEpochs,1));
Lhard=ceH+cfg.loss.kldWeightHard*klH;
Lweak=ceS+(lambda*cfg.loss.kldWeightWeakMax)*klS;

loss=Lhard+cfg.loss.betaWeak*Lweak;
end
