function makePlots(R,Hb,Hp,cfg)
%MAKEPLOTS Save comparison and representative time-history plots.
figdir=cfg.paths.figures; if ~exist(figdir,'dir'), mkdir(figdir); end

f=figure('Name','Training curves'); tiledlayout(2,1);
nexttile; plot(Hb.return,'LineWidth',1.2); hold on; plot(Hp.return,'LineWidth',1.2); grid on;
xlabel('Episode'); ylabel('Return'); legend('Manual','Ontology-RGAT','Location','best'); title('PPO training return');
nexttile; plot(movmean(Hb.success,max(1,round(numel(Hb.success)/10))),'LineWidth',1.2); hold on;
plot(movmean(Hp.success,max(1,round(numel(Hp.success)/10))),'LineWidth',1.2); grid on; ylim([0 1]);
xlabel('Episode'); ylabel('Moving success rate'); legend('Manual','Ontology-RGAT','Location','best');
exportgraphics(f,fullfile(figdir,'training_curves.png'),'Resolution',180);

f=figure('Name','Evaluation summary'); tiledlayout(2,2);
nexttile; bar(100*R.summary.SuccessRate); set(gca,'XTickLabel',R.summary.Policy); ylabel('Success [%]'); ylim([0 100]); grid on; title('Landing success');
nexttile; bar(R.summary.MeanTouchdownXY); set(gca,'XTickLabel',R.summary.Policy); ylabel('m'); grid on; title('Touchdown XY error');
nexttile; bar(R.summary.MeanMaxTiltDeg); set(gca,'XTickLabel',R.summary.Policy); ylabel('deg'); grid on; title('Maximum tilt');
nexttile; bar(R.summary.MeanEnergyJ); set(gca,'XTickLabel',R.summary.Policy); ylabel('J (ideal induced-power proxy)'); grid on; title('Energy proxy');
exportgraphics(f,fullfile(figdir,'evaluation_summary.png'),'Resolution',180);

Lb=R.baseline.representative; Lp=R.proposed.representative;
f=figure('Name','Representative 3D trajectories');
plot3(Lb.x(1,:),Lb.x(2,:),Lb.x(3,:),'LineWidth',1.4); hold on;
plot3(Lp.x(1,:),Lp.x(2,:),Lp.x(3,:),'LineWidth',1.4); plot3(0,0,cfg.sim.groundZ,'o','MarkerSize',8);
grid on; axis equal; xlabel('East x [m]'); ylabel('North y [m]'); zlabel('Up z [m]');
legend('Manual','Ontology-RGAT','Pad','Location','best'); title('Representative landing trajectories');
exportgraphics(f,fullfile(figdir,'representative_trajectory_3d.png'),'Resolution',180);

f=figure('Name','Representative disturbance response'); tiledlayout(3,1);
nexttile; plot(Lp.t,Lp.wind','LineWidth',1.0); grid on; ylabel('Wind [m/s]'); legend('u','v','w'); title('Panel-averaged local wind');
nexttile; plot(Lp.t,Lp.aeroF,'LineWidth',1.2); grid on; ylabel('|F_a| [N]'); title('Distributed aerodynamic resultant');
nexttile; plot(Lp.t,rad2deg(Lp.tilt),'LineWidth',1.2); grid on; ylabel('Tilt [deg]'); xlabel('Time [s]'); title('Attitude response');
exportgraphics(f,fullfile(figdir,'proposed_disturbance_response.png'),'Resolution',180);

% Learned relation-attention summary over the representative proposed rollout.
relAcc=zeros(1,cfg.ontology.nRelations); nRelSamples=0;
for k=1:size(Lp.graphX,3)
    if mod(k,5)~=1, continue; end
    g=semantic.buildOntologyGraph(struct('positionError',0,'verticalSpeed',0,'tilt',0,'angularRate',0,'windRisk',0,'markerQuality',0,'visualStability',0,'alignment',0,'attitudeStability',0,'touchdownSafety',0),cfg);
    g.X=Lp.graphX(:,:,k);
    e=rgat.explain(R.proposedModel,g);
    relAcc=relAcc+e.relationMean; nRelSamples=nRelSamples+1;
end
if nRelSamples>0
    relAcc=relAcc/nRelSamples;
    f=figure('Name','R-GAT relation attention');
    bar(relAcc); set(gca,'XTick',1:cfg.ontology.nRelations,'XTickLabel',cfg.ontology.relationNames); ylabel('Mean attention'); grid on;
    title('Proposed R-GAT: mean relation attention (representative rollout)');
    exportgraphics(f,fullfile(figdir,'rgat_relation_attention.png'),'Resolution',180);
end

% Instantaneous distributed panel loads at the strongest aerodynamic-load time.
[~,kp]=max(Lp.aeroF);
f=viz.plotSurfaceLoads(Lp.x(:,kp),Lp.t(kp),cfg);
exportgraphics(f,fullfile(figdir,'panel_surface_loads_peak.png'),'Resolution',180);


if isfield(R,'windSweep') && ~isempty(R.windSweep)
    f=figure('Name','Wind generalization sweep'); hold on;
    labels=unique(R.windSweep.Policy,'stable');
    for jj=1:numel(labels)
        ix=R.windSweep.Policy==labels(jj);
        plot(R.windSweep.WindScale(ix),100*R.windSweep.SuccessRate(ix),'-o','LineWidth',1.3);
    end
    grid on; ylim([0 100]); xlabel('Wind intensity scale'); ylabel('Success [%]');
    legend(cellstr(labels),'Location','best'); title('Generalization to disturbance intensity');
    exportgraphics(f,fullfile(figdir,'wind_generalization_success.png'),'Resolution',180);
end

end
