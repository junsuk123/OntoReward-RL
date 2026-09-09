function makePlots(R,Hb,Hp,cfg)
%MAKEPLOTS External-simulator publication plots; no MATLAB aero recomputation.
figdir=cfg.paths.figures; if ~isfolder(figdir), mkdir(figdir); end

f=figure('Name','Training curves'); tiledlayout(2,1);
nexttile; plot(Hb.return,'LineWidth',1.2); hold on;
plot(Hp.return,'LineWidth',1.2); grid on;
xlabel('Episode'); ylabel('Return'); legend('Manual','Ontology-RGAT','Location','best');
title('PPO training return (Isaac/PX4)');
nexttile; plot(movmean(Hb.success,max(1,round(numel(Hb.success)/10))),'LineWidth',1.2); hold on;
plot(movmean(Hp.success,max(1,round(numel(Hp.success)/10))),'LineWidth',1.2);
grid on; ylim([0 1]); xlabel('Episode'); ylabel('Moving success rate');
legend('Manual','Ontology-RGAT','Location','best');
exportgraphics(f,fullfile(figdir,'training_curves.png'),'Resolution',180);

f=figure('Name','Evaluation summary'); tiledlayout(2,2);
nexttile; bar(100*R.summary.SuccessRate); set(gca,'XTickLabel',R.summary.Policy);
ylabel('Success [%]'); ylim([0 100]); grid on; title('Landing success');
nexttile; bar(R.summary.MeanTouchdownXY); set(gca,'XTickLabel',R.summary.Policy);
ylabel('m'); grid on; title('Touchdown XY error');
nexttile; bar(R.summary.MeanMaxTiltDeg); set(gca,'XTickLabel',R.summary.Policy);
ylabel('deg'); grid on; title('Maximum tilt');
nexttile; bar(R.summary.MeanMaxAeroForce); set(gca,'XTickLabel',R.summary.Policy);
ylabel('N'); grid on; title('Peak Isaac aerodynamic force');
exportgraphics(f,fullfile(figdir,'evaluation_summary.png'),'Resolution',180);

Lb=R.baseline.representative; Lp=R.proposed.representative;
f=figure('Name','Representative 3D trajectories');
plot3(Lb.x(1,:),Lb.x(2,:),Lb.x(3,:),'LineWidth',1.4); hold on;
plot3(Lp.x(1,:),Lp.x(2,:),Lp.x(3,:),'LineWidth',1.4);
plot3(0,0,cfg.sim.groundZ,'o','MarkerSize',8); grid on; axis equal;
xlabel('East x [m]'); ylabel('North y [m]'); zlabel('Up z [m]');
legend('Manual','Ontology-RGAT','Pad','Location','best');
title('Representative Isaac/PX4 landing trajectories');
exportgraphics(f,fullfile(figdir,'representative_trajectory_3d.png'),'Resolution',180);

f=figure('Name','Representative disturbance response'); tiledlayout(3,1);
nexttile; plot(Lp.t,Lp.wind','LineWidth',1.0); grid on; ylabel('Wind [m/s]');
legend('East','North','Up'); title('Isaac wind field');
nexttile; plot(Lp.t,Lp.aeroF,'LineWidth',1.2); grid on;
ylabel('|F_a| [N]'); title('Isaac aerodynamic resultant');
nexttile; plot(Lp.t,rad2deg(Lp.tilt),'LineWidth',1.2); grid on;
ylabel('Tilt [deg]'); xlabel('Time [s]'); title('PX4 attitude response');
exportgraphics(f,fullfile(figdir,'proposed_disturbance_response.png'),'Resolution',180);

relAcc=zeros(1,cfg.ontology.nRelations); nRelSamples=0;
for k=1:size(Lp.graphX,3)
    if mod(k,5)~=1, continue; end
    g=semantic.buildOntologyGraph(struct('positionError',0,'verticalSpeed',0, ...
        'tilt',0,'angularRate',0,'windRisk',0,'markerQuality',0, ...
        'visualStability',0,'alignment',0,'attitudeStability',0, ...
        'touchdownSafety',0),cfg);
    g.X=Lp.graphX(:,:,k); e=rgat.explain(R.proposedModel,g);
    relAcc=relAcc+e.relationMean; nRelSamples=nRelSamples+1;
end
if nRelSamples>0
    f=figure('Name','R-GAT relation attention');
    bar(relAcc/nRelSamples); set(gca,'XTick',1:cfg.ontology.nRelations, ...
        'XTickLabel',cfg.ontology.relationNames); ylabel('Mean attention'); grid on;
    title('R-GAT relation attention on Isaac/PX4 rollout');
    exportgraphics(f,fullfile(figdir,'rgat_relation_attention.png'),'Resolution',180);
end

[~,kp]=max(Lp.aeroF); p=Lp.x(1:3,kp); force=Lp.aeroForce(:,kp);
f=figure('Name','Peak external aerodynamic load');
plot3(Lp.x(1,:),Lp.x(2,:),Lp.x(3,:),'Color',[0.75 0.75 0.75]); hold on;
scatter3(p(1),p(2),p(3),50,Lp.aeroF(kp),'filled');
quiver3(p(1),p(2),p(3),force(1),force(2),force(3),0.8,'LineWidth',1.5);
grid on; axis equal; xlabel('East [m]'); ylabel('North [m]'); zlabel('Up [m]');
title(sprintf('Isaac resultant aerodynamic load at t=%.2f s',Lp.t(kp))); colorbar;
exportgraphics(f,fullfile(figdir,'external_aero_load_peak.png'),'Resolution',180);

if isfield(R,'windSweep') && ~isempty(R.windSweep)
    f=figure('Name','Wind generalization sweep'); hold on;
    labels=unique(R.windSweep.Policy,'stable');
    for j=1:numel(labels)
        ix=R.windSweep.Policy==labels(j);
        plot(R.windSweep.WindScale(ix),100*R.windSweep.SuccessRate(ix),'-o','LineWidth',1.3);
    end
    grid on; ylim([0 100]); xlabel('Isaac wind intensity scale'); ylabel('Success [%]');
    legend(cellstr(labels),'Location','best'); title('External disturbance generalization');
    exportgraphics(f,fullfile(figdir,'wind_generalization_success.png'),'Resolution',180);
end
end
