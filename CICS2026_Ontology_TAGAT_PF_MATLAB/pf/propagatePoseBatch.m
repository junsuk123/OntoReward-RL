function out = propagatePoseBatch(p,d)
%PROPAGATEPOSEBATCH Apply body-frame odometry delta to particle poses.

c=cos(p(:,3)); s=sin(p(:,3));
out = p;
out(:,1)=p(:,1)+c*d(1)-s*d(2);
out(:,2)=p(:,2)+s*d(1)+c*d(2);
out(:,3)=wrapAngle(p(:,3)+d(3));
end
