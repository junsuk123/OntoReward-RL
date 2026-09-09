function f = plotSurfaceLoads(x,t,cfg)
%PLOTSURFACELOADS Visualize panel locations and instantaneous aerodynamic forces.
d=aero.distributedAero(x,t,cfg); R=mathx.quatToRotm(x(7:10));
P=cfg.aero.panels; C=zeros(3,P.count); F=zeros(3,P.count);
for i=1:P.count
    C(:,i)=x(1:3)+R*P.centerB(:,i); F(:,i)=R*d.panelForceB(:,i);
end
f=figure('Name','Distributed surface loads');
scatter3(C(1,:),C(2,:),C(3,:),20,vecnorm(F),'filled'); hold on;
quiver3(C(1,:),C(2,:),C(3,:),F(1,:),F(2,:),F(3,:),0.8);
axis equal; grid on; xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title(sprintf('Panel aerodynamic loads at t=%.2f s',t)); colorbar;
end
