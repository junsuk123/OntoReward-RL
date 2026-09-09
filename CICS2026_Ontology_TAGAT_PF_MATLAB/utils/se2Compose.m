function c = se2Compose(a,b)
%SE2COMPOSE c = a ⊕ b, with a and b = [x y yaw].
ca = cos(a(3)); sa = sin(a(3));
c = [a(1) + ca*b(1) - sa*b(2), ...
     a(2) + sa*b(1) + ca*b(2), ...
     wrapAngle(a(3)+b(3))];
end
