function plot_comparison(y,pBase,pTAGAT,uBase,uTAGAT,cfg)
x=1:numel(y);

f=figure('Color','w','Position',[100 100 1400 720]);
tiledlayout(3,1,'TileSpacing','compact');

nexttile;
stairs(x,y,'k','LineWidth',1.1); ylim([-0.05 1.05]);
ylabel('GT Fault'); grid on;
title('Deep Urban Generalization Test');

nexttile;
plot(x,pBase,'LineWidth',1.0); hold on;
plot(x,pTAGAT,'LineWidth',1.0);
yline(0.5,'--');
ylim([0 1]); grid on;
ylabel('Fault Probability');
legend('Transformer-Dual EDL','Ontology-TA-GAT','Threshold','Location','best');

nexttile;
plot(x,uBase,'LineWidth',1.0); hold on;
plot(x,uTAGAT,'LineWidth',1.0);
ylim([0 1]); grid on;
ylabel('EDL Uncertainty'); xlabel('Window index');
legend('Transformer-Dual EDL','Ontology-TA-GAT','Location','best');

exportgraphics(f,fullfile(cfg.resultsDir,'fault_probability_comparison.png'),'Resolution',160);

% Confusions
figure('Color','w');
confusionchart(categorical(y),categorical(pBase>=0.5));
title('Transformer-Dual EDL');
exportgraphics(gcf,fullfile(cfg.resultsDir,'confusion_transformer.png'),'Resolution',160);

figure('Color','w');
confusionchart(categorical(y),categorical(pTAGAT>=0.5));
title('Ontology-TA-GAT + Dual EDL');
exportgraphics(gcf,fullfile(cfg.resultsDir,'confusion_tagat.png'),'Resolution',160);
end
