function q = quatNormalize(q)
q = q(:);
n = norm(q);
if n < 1e-12, q = [1;0;0;0]; else, q = q/n; end
if q(1) < 0, q = -q; end
end
