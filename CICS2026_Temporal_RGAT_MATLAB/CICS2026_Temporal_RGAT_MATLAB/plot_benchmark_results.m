function plot_benchmark_results(results,Y,cfg)
%PLOT_BENCHMARK_RESULTS Figures appear immediately and are also exported.
keys=fieldnames(results); n=numel(keys);
names=strings(n,1); f1=zeros(n,1); au=zeros(n,1);
for i=1:n
    names(i)=results.(keys{i}).name;
    f1(i)=results.(keys{i}).metrics.F1;
    au(i)=results.(keys{i}).metrics.AUROC;
end
% Figures travel out of their directory into slides, so they have to carry their
% own provenance: which supervision, and which evaluation set.
if cfg.allowSurrogateSoftLabel
    tag=" [NON-STRICT: "+cfg.weakLabel.source+" weak label]";
else
    tag="";
end
if cfg.split.mode=="cross_scenario"
    setName="Deep Urban (unseen scenario)";
else
    setName="Pooled chronological test set";
end

figure('Name','CICS2026 Benchmark Summary','Color','w');
tiledlayout(1,2);
nexttile; bar(f1); ylim([0 1]); xticks(1:n); xticklabels(names); xtickangle(45);
ylabel('F1'); title(setName+" F1"+tag); grid on;
nexttile; bar(au); ylim([0 1]); xticks(1:n); xticklabels(names); xtickangle(45);
ylabel('AUROC'); title(setName+" AUROC"+tag); grid on;
exportgraphics(gcf,fullfile(cfg.outputDir,'benchmark_summary.png'),'Resolution',180);

prior=matlab.lang.makeValidName('Transformer-DualEDL');
prop=matlab.lang.makeValidName('TemporalR-GAT-DualEDL');
thrPrior=thr_of(results.(prior),cfg); thrProp=thr_of(results.(prop),cfg);
figure('Name','Fault Probability','Color','w');
plot(Y,'k','LineWidth',1.2); hold on;
hPrior=plot(results.(prior).pFault,'LineWidth',1.0);
hProp=plot(results.(prop).pFault,'LineWidth',1.0);
% One line per model: each operates at its own validation-selected threshold, so
% a single shared 0.5 line would misrepresent where either actually decides.
yline(thrPrior,'--','Color',hPrior.Color);
yline(thrProp,'--','Color',hProp.Color);
ylim([-0.05 1.05]); grid on;
legend('Hard label', ...
    sprintf('Prior Transformer-Dual EDL (thr %.2f)',thrPrior), ...
    sprintf('Temporal R-GAT-Dual EDL (thr %.2f)',thrProp), ...
    'Prior threshold','Proposed threshold','Location','best');
xlabel('Window'); ylabel('Fault probability'); title(setName+tag);
exportgraphics(gcf,fullfile(cfg.outputDir,'fault_probability_timeseries.png'),'Resolution',180);
end

function t=thr_of(R,cfg)
if isfield(R,'threshold') && ~isempty(R.threshold), t=double(R.threshold); else, t=cfg.threshold; end
end
