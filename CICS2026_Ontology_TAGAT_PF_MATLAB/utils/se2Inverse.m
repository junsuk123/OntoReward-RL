function b = se2Inverse(a)
c = cos(a(3)); s = sin(a(3));
b = [-c*a(1)-s*a(2), s*a(1)-c*a(2), wrapAngle(-a(3))];
end
