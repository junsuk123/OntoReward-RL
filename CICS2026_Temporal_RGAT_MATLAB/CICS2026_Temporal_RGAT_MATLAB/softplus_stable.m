function y=softplus_stable(x)
%SOFTPLUS_STABLE Numerically stable softplus.
y=max(x,0)+log(1+exp(-abs(x)));
end
