function [D,mu,sigma] = normalize_scenarios(D,cfg)
%NORMALIZE_SCENARIOS Fit normalization only on training scenarios.
Xfit=[D.medium.X;D.harsh.X];
mu=mean(Xfit,1,'omitnan'); sigma=std(Xfit,0,1,'omitnan');
sigma(sigma<1e-8)=1;
for s=["medium","harsh","deep"]
    D.(char(s)).X=single((D.(char(s)).X-mu)./sigma);
end
end
