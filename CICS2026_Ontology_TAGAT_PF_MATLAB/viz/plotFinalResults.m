function plotFinalResults(data,testIdx,results,pred,G,metrics,cfg,runDir)
%PLOTFINALRESULTS Final cross-baseline figures.

names=string(metrics.Model);
keys=arrayfun(@matlab.lang.makeValidName,names,"UniformOutput",false);
tt=data.time(testIdx)-data.time(testIdx(1));

% 1) trajectory
f=figure("Color","w","Position",[100 100 1050 760]);
plot(data.gt(testIdx,1),data.gt(testIdx,2),"k-","LineWidth",2); hold on;
for k=1:numel(keys)
    r=results.(keys{k});
    plot(r.est(:,1),r.est(:,2),"LineWidth",1.1);
end
axis equal; grid on; xlabel("x [m]"); ylabel("y [m]");
title("UGV localization trajectory comparison");
legend(["Ground truth";names(:)],"Interpreter","none","Location","bestoutside");
saveFigure(f,runDir,"final_trajectory",cfg);

% 2) position errors
f=figure("Color","w","Position",[100 100 1100 650]); hold on;
for k=1:numel(keys)
    r=results.(keys{k});
    e=sqrt(sum((r.est(:,1:2)-data.gt(testIdx,1:2)).^2,2));
    plot(tt,e,"LineWidth",1.0);
end
grid on; xlabel("time [s]"); ylabel("position error [m]");
title("Position error over test segment");
legend(names,"Interpreter","none","Location","bestoutside");
saveFigure(f,runDir,"final_position_error",cfg);

% 3) accuracy summary
f=figure("Color","w","Position",[100 100 1100 620]);
tl=tiledlayout(1,2,"Padding","compact","TileSpacing","compact");
ax=nexttile(tl); bar(ax,metrics.PositionRMSE_m); grid(ax,"on");
ax.XTick=1:height(metrics); ax.XTickLabel=names; ax.XTickLabelRotation=25;
ylabel(ax,"RMSE [m]"); title(ax,"Position RMSE");
ax=nexttile(tl); bar(ax,metrics.PositionP95_m); grid(ax,"on");
ax.XTick=1:height(metrics); ax.XTickLabel=names; ax.XTickLabelRotation=25;
ylabel(ax,"P95 error [m]"); title(ax,"95th percentile position error");
saveFigure(f,runDir,"final_accuracy_summary",cfg);

% 4) particle and runtime summary
f=figure("Color","w","Position",[100 100 1100 620]);
tl=tiledlayout(1,2,"Padding","compact","TileSpacing","compact");
ax=nexttile(tl); bar(ax,metrics.MeanParticles); grid(ax,"on");
ax.XTick=1:height(metrics); ax.XTickLabel=names; ax.XTickLabelRotation=25;
ylabel(ax,"mean particles"); title(ax,"Sampling budget");
ax=nexttile(tl); bar(ax,metrics.MeanRuntime_ms); grid(ax,"on");
ax.XTick=1:height(metrics); ax.XTickLabel=names; ax.XTickLabelRotation=25;
ylabel(ax,"mean runtime [ms/step]"); title(ax,"Computational cost");
saveFigure(f,runDir,"final_particles_runtime",cfg);

% 5) reliability traces for proposed if available.
propKey=matlab.lang.makeValidName("Ontology-TAGAT-PF");
if isfield(results,propKey)
    r=results.(propKey);
    f=figure("Color","w","Position",[100 100 1100 660]);
    plot(tt,r.reliability,"LineWidth",1.0); grid on; ylim([0 1]);
    xlabel("time [s]"); ylabel("score");
    legend(["GNSS","LiDAR","Odometry","Difficulty"],"Location","bestoutside");
    title("TA-GAT reliability / localization difficulty");
    saveFigure(f,runDir,"final_reliability",cfg);
end

% 6) top TA-GAT edge attention.
if isfield(pred,"TAGAT") && ~isempty(pred.TAGAT.edgeAttention)
    A=pred.TAGAT.edgeAttention(:,testIdx);
    ma=mean(A,2,"omitnan");
    [~,ord]=sort(ma,"descend");
    ord=ord(1:min(6,numel(ord)));

    f=figure("Color","w","Position",[100 100 1150 680]); hold on;
    labels=strings(numel(ord),1);
    for z=1:numel(ord)
        plot(tt,A(ord(z),:),"LineWidth",1.0);
        e=ord(z);
        labels(z)=G.nodeNames(G.src(e))+" -"+G.relationNames(G.rel(e))+"-> "+G.nodeNames(G.dst(e));
    end
    grid on; xlabel("time [s]"); ylabel("edge attention");
    title("Top time-varying ontology edge attention (TA-GAT)");
    legend(labels,"Interpreter","none","Location","bestoutside");
    saveFigure(f,runDir,"final_TAGAT_attention",cfg);
end
end

function saveFigure(f,runDir,name,cfg)
exportgraphics(f,fullfile(runDir,name+".png"),"Resolution",170);
if cfg.visual.saveFig
    savefig(f,fullfile(runDir,name+".fig"));
end
end
