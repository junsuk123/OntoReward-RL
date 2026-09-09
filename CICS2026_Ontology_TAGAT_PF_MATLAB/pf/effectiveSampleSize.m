function ess = effectiveSampleSize(w)
w=w(:); w=w/sum(w);
ess=1/sum(w.^2);
end
