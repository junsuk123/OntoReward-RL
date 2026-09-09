function rpy = quatToEulerZYX(q)
R = mathx.quatToRotm(q);
% Robust ZYX extraction.
pitch = asin(max(-1,min(1,-R(3,1))));
if abs(cos(pitch)) > 1e-8
    roll = atan2(R(3,2),R(3,3));
    yaw  = atan2(R(2,1),R(1,1));
else
    roll = 0;
    yaw = atan2(-R(1,2),R(2,2));
end
rpy = [roll;pitch;yaw];
end
