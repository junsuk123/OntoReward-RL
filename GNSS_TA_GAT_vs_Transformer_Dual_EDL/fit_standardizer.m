function [mu,sigma] = fit_standardizer(seq)
X = [];
for i=1:numel(seq)
    X = [X; double(seq{i}.X)]; %#ok<AGROW>
end
mu = mean(X,1,'omitnan');
sigma = std(X,0,1,'omitnan');
sigma(sigma < 1e-8) = 1;
mu = single(mu);
sigma = single(sigma);
end
