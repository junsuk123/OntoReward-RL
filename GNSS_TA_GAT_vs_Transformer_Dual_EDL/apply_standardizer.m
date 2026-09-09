function seq = apply_standardizer(seq,mu,sigma)
for i=1:numel(seq)
    seq{i}.X = (seq{i}.X - mu) ./ sigma;
end
end
