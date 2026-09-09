function [mu,P] = weightedPoseStats(p,w)
w=w(:)/sum(w);
x=sum(w.*p(:,1));
y=sum(w.*p(:,2));
c=sum(w.*cos(p(:,3)));
s=sum(w.*sin(p(:,3)));
yaw=atan2(s,c);
mu=[x y yaw];

e=[p(:,1)-x,p(:,2)-y,wrapAngle(p(:,3)-yaw)];
P=(e.*w).'*e + 1e-9*eye(3);
end
