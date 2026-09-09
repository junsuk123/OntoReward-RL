function H = initLiveVisualizer(data,testIdx,name,cfg) %#ok<INUSD>
%INITLIVEVISUALIZER Four-panel live diagnostic figure.

H.fig=figure("Name","Live - "+name,"Color","w","Position",[80 80 1200 780]);
tl=tiledlayout(H.fig,2,2,"Padding","compact","TileSpacing","compact");

H.axTraj=nexttile(tl,1);
plot(H.axTraj,data.gt(testIdx,1),data.gt(testIdx,2),"k-","LineWidth",1.4); hold(H.axTraj,"on");
H.estLine=plot(H.axTraj,nan,nan,"-","LineWidth",1.4);
H.particlePts=scatter(H.axTraj,nan,nan,7,"filled","MarkerFaceAlpha",0.18);
axis(H.axTraj,"equal"); grid(H.axTraj,"on");
title(H.axTraj,"Trajectory / particles"); xlabel(H.axTraj,"x [m]"); ylabel(H.axTraj,"y [m]");

H.axErr=nexttile(tl,2);
H.errLine=plot(H.axErr,nan,nan,"LineWidth",1.2); grid(H.axErr,"on");
title(H.axErr,"Position error"); xlabel(H.axErr,"time [s]"); ylabel(H.axErr,"error [m]");

H.axPF=nexttile(tl,3);
yyaxis(H.axPF,"left");
H.essLine=plot(H.axPF,nan,nan,"LineWidth",1.2); ylabel(H.axPF,"ESS");
yyaxis(H.axPF,"right");
H.nLine=plot(H.axPF,nan,nan,"LineWidth",1.2); ylabel(H.axPF,"particles");
grid(H.axPF,"on"); title(H.axPF,"PF health"); xlabel(H.axPF,"time [s]");

H.axRel=nexttile(tl,4);
H.relBars=bar(H.axRel,1:4,[0 0 0 0]);
H.axRel.XTick=1:4; H.axRel.XTickLabel={"GNSS","LiDAR","Odom","Difficulty"};
ylim(H.axRel,[0 1]); grid(H.axRel,"on"); title(H.axRel,"Current reliability / difficulty");

title(tl,name,"Interpreter","none");
end
