function q = eulerToQuat(rpy)
% ZYX Euler [roll pitch yaw] -> scalar-first quaternion.
r=rpy(1); p=rpy(2); y=rpy(3);
cr=cos(r/2); sr=sin(r/2); cp=cos(p/2); sp=sin(p/2); cy=cos(y/2); sy=sin(y/2);
q = [cr*cp*cy + sr*sp*sy; ...
     sr*cp*cy - cr*sp*sy; ...
     cr*sp*cy + sr*cp*sy; ...
     cr*cp*sy - sr*sp*cy];
q = mathx.quatNormalize(q);
end
