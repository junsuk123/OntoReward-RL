classdef RealtimeMonitor < handle
    properties
        Fig; TL; Ax3D; Traj; ForceQuiver; Pos; Wind; Tilt; Reward; Aero; Cfg;
    end
    methods
        function obj=RealtimeMonitor(cfg)
            obj.Cfg=cfg;
            obj.Fig=figure('Name','UAV landing real-time monitor','NumberTitle','off');
            obj.TL=tiledlayout(obj.Fig,2,3);
            ax=nexttile; obj.Ax3D=ax; obj.Traj=plot3(ax,nan,nan,nan,'LineWidth',1.4); hold(ax,'on');
            plot3(ax,0,0,cfg.sim.groundZ,'o');
            obj.ForceQuiver=quiver3(ax,nan,nan,nan,nan,nan,nan,0.7);
            grid(ax,'on'); axis(ax,'equal'); xlabel(ax,'x'); ylabel(ax,'y'); zlabel(ax,'z'); title(ax,'Trajectory + panel aero loads');
            ax=nexttile; obj.Pos=plot(ax,nan,nan,'LineWidth',1.2); grid(ax,'on'); title(ax,'XY error'); xlabel(ax,'s'); ylabel(ax,'m');
            ax=nexttile; hold(ax,'on'); obj.Wind=gobjects(3,1); for i=1:3, obj.Wind(i)=plot(ax,nan,nan,'LineWidth',1.0); end; grid(ax,'on'); title(ax,'Mean local wind'); xlabel(ax,'s'); ylabel(ax,'m/s'); legend(ax,{'u','v','w'});
            ax=nexttile; obj.Tilt=plot(ax,nan,nan,'LineWidth',1.2); grid(ax,'on'); title(ax,'Tilt'); xlabel(ax,'s'); ylabel(ax,'deg');
            ax=nexttile; obj.Reward=plot(ax,nan,nan,'LineWidth',1.2); grid(ax,'on'); title(ax,'Reward'); xlabel(ax,'s'); ylabel(ax,'r_t');
            ax=nexttile; obj.Aero=plot(ax,nan,nan,'LineWidth',1.2); grid(ax,'on'); title(ax,'Aerodynamic load'); xlabel(ax,'s'); ylabel(ax,'N');
        end
        function update(obj,L,k,env,cur,info)
            idx=1:k; set(obj.Traj,'XData',L.x(1,idx),'YData',L.x(2,idx),'ZData',L.x(3,idx));
            set(obj.Pos,'XData',L.t(idx),'YData',vecnorm(L.x(1:2,idx)));
            for i=1:3, set(obj.Wind(i),'XData',L.t(idx),'YData',L.wind(i,idx)); end
            set(obj.Tilt,'XData',L.t(idx),'YData',rad2deg(L.tilt(idx)));
            set(obj.Reward,'XData',L.t(idx),'YData',L.r(idx)); set(obj.Aero,'XData',L.t(idx),'YData',L.aeroF(idx));
            % Current distributed surface forces in inertial coordinates.
            P=obj.Cfg.aero.panels; R=mathx.quatToRotm(env.x(7:10)); C=zeros(3,P.count); F=zeros(3,P.count);
            for j=1:P.count
                C(:,j)=env.x(1:3)+R*P.centerB(:,j);
                F(:,j)=R*info.diag.aero.panelForceB(:,j);
            end
            set(obj.ForceQuiver,'XData',C(1,:),'YData',C(2,:),'ZData',C(3,:), ...
                'UData',F(1,:),'VData',F(2,:),'WData',F(3,:));
            title(obj.TL,sprintf('t=%.2f s | %s | wind risk %.2f | |F_a| %.2f N', ...
                env.t,info.status,cur.sem.windRisk,info.diag.aeroForceMag));
        end
    end
end
