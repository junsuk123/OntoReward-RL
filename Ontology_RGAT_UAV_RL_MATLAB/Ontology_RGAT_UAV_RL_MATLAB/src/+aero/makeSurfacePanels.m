function panels = makeSurfacePanels(cfg)
%MAKESURFACEPANELS Panelized fuselage + four arm boxes in body FLU frame.
C=[]; N=[]; A=[]; Lc=[]; names={};
    function addBox(center,dims,nGrid,label,Rbox)
        hx=dims(1)/2; hy=dims(2)/2; hz=dims(3)/2;
        faces = { [1;0;0], [hx,0,0], [2,3], [dims(2),dims(3)]; ...
                 [-1;0;0],[-hx,0,0],[2,3],[dims(2),dims(3)]; ...
                 [0;1;0], [0,hy,0], [1,3], [dims(1),dims(3)]; ...
                 [0;-1;0],[0,-hy,0],[1,3],[dims(1),dims(3)]; ...
                 [0;0;1], [0,0,hz], [1,2], [dims(1),dims(2)]; ...
                 [0;0;-1],[0,0,-hz],[1,2],[dims(1),dims(2)]};
        for f=1:size(faces,1)
            n=faces{f,1}; off=faces{f,2}; axesIdx=faces{f,3}; lens=faces{f,4};
            n1=nGrid; n2=nGrid;
            u=((1:n1)-0.5)/n1-0.5; v=((1:n2)-0.5)/n2-0.5;
            for ii=1:n1
                for jj=1:n2
                    local=[0;0;0]; local(axesIdx(1))=u(ii)*lens(1); local(axesIdx(2))=v(jj)*lens(2);
                    c=center + Rbox*(off(:) + local);
                    C(:,end+1)=c; %#ok<AGROW>
                    N(:,end+1)=Rbox*n; %#ok<AGROW>
                    A(end+1)=lens(1)*lens(2)/(n1*n2); %#ok<AGROW>
                    Lc(end+1)=sqrt(A(end)); %#ok<AGROW>
                    names{end+1}=sprintf('%s_f%d',label,f); %#ok<AGROW>
                end
            end
        end
    end
addBox([0;0;0],cfg.drone.bodyDims,cfg.aero.panelGrid,'fuselage',eye(3));
if cfg.aero.includeArms
    L=cfg.drone.armLength;
    armLen = max(0.05,L-0.10);
    w=cfg.drone.armWidth; h=cfg.drone.armHeight;
    centers = [ L/2, -L/2, -L/2, L/2; L/2, L/2, -L/2, -L/2; 0,0,0,0];
    for k=1:4
        % Axis-aligned box approximation. This intentionally over-resolves surface drag
        % rather than using one global drag coefficient.
        yaw=atan2(centers(2,k),centers(1,k)); Rz=[cos(yaw) -sin(yaw) 0; sin(yaw) cos(yaw) 0; 0 0 1];
        addBox(centers(:,k),[armLen,w,h],cfg.aero.armPanelGrid,sprintf('arm%d',k),Rz);
    end
end
panels.centerB=C; panels.normalB=N; panels.area=A; panels.charLength=Lc; panels.name=names;
panels.count=size(C,2);
end
