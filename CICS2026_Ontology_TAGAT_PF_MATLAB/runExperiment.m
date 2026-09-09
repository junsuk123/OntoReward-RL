function summary = runExperiment(cfg)
%RUNEXPERIMENT End-to-end benchmark runner.

rng(cfg.seed, "twister");

runStamp = char(datetime("now","Format","yyyyMMdd_HHmmss"));
runDir = fullfile(cfg.paths.results, "run_" + string(runStamp));
if ~exist(runDir,"dir"), mkdir(runDir); end

diary(fullfile(runDir,"console.log"));
cleanupObj = onCleanup(@() diary("off")); %#ok<NASGU>

fprintf("\n=== CICS2026 Ontology-TA-GAT Adaptive PF Experiment ===\n");
fprintf("Mode        : %s\n", cfg.data.mode);
fprintf("Run folder  : %s\n", runDir);
fprintf("MATLAB      : %s\n\n", version);

saveConfig(cfg, fullfile(runDir,"config.json"));

switch upper(cfg.data.mode)
    case "NCLT"
        data = prepareNCLT(cfg);
    case "SYNTHETIC"
        data = generateSyntheticDemo(cfg);
    otherwise
        error("Unknown cfg.data.mode: %s", cfg.data.mode);
end

T = numel(data.time);
[trainIdx,valIdx,testIdx] = splitIndices(T,cfg);

fprintf("Samples: total=%d train=%d val=%d test=%d\n", ...
    T,numel(trainIdx),numel(valIdx),numel(testIdx));

ontologyGraph = buildOntologyGraph();
validateOntologyGraph(ontologyGraph);

graphPack = buildGraphDataset(data,trainIdx,cfg);

datasetSummary = summarizeDataset(data);
writetable(datasetSummary,fullfile(runDir,"dataset_summary.csv"));

% dlgradient is only resolvable inside dlfeval, so exist("dlgradient","file")
% returns 0 even when Deep Learning Toolbox is installed. Probe the class and
% dlfeval instead.
haveDL = exist("dlarray","class") == 8 && exist("dlfeval","file") == 2 && ...
    license("test","Neural_Network_Toolbox") == 1;
models = struct();
pred = struct();

if haveDL
    fprintf("\n--- Training GAT baseline ---\n");
    gatGraph = makeGraphVariant(ontologyGraph,"GAT");
    [models.GAT,histG] = trainGraphModel(graphPack,gatGraph,"GAT",trainIdx,valIdx,cfg);
    writetable(histG,fullfile(runDir,"training_GAT.csv"));
    saveModelPortable(models.GAT,fullfile(runDir,"model_GAT.mat"));
    pred.GAT = predictGraphModel(models.GAT,graphPack,gatGraph,"GAT",cfg);

    fprintf("\n--- Training ontology R-GAT baseline ---\n");
    [models.RGAT,histR] = trainGraphModel(graphPack,ontologyGraph,"RGAT",trainIdx,valIdx,cfg);
    writetable(histR,fullfile(runDir,"training_RGAT.csv"));
    saveModelPortable(models.RGAT,fullfile(runDir,"model_RGAT.mat"));
    pred.RGAT = predictGraphModel(models.RGAT,graphPack,ontologyGraph,"RGAT",cfg);

    fprintf("\n--- Training ontology TA-GAT proposed model ---\n");
    [models.TAGAT,histT] = trainGraphModel(graphPack,ontologyGraph,"TAGAT",trainIdx,valIdx,cfg);
    writetable(histT,fullfile(runDir,"training_TAGAT.csv"));
    saveModelPortable(models.TAGAT,fullfile(runDir,"model_TAGAT.mat"));
    pred.TAGAT = predictGraphModel(models.TAGAT,graphPack,ontologyGraph,"TAGAT",cfg);

    save(fullfile(runDir,"predictions.mat"),"pred","-v7.3");

    saveAttentionSummary(pred.GAT,gatGraph,fullfile(runDir,"attention_GAT.csv"),testIdx);
    saveAttentionSummary(pred.RGAT,ontologyGraph,fullfile(runDir,"attention_RGAT.csv"),testIdx);
    saveAttentionSummary(pred.TAGAT,ontologyGraph,fullfile(runDir,"attention_TAGAT.csv"),testIdx);
else
    warning("Deep Learning Toolbox functions were not found. Learned graph baselines are skipped.");
end

baselineNames = ["PF-Fixed","PF-Adaptive","Ontology-Rule-PF"];
if haveDL
    baselineNames = [baselineNames,"GAT-PF","Ontology-RGAT-PF","Ontology-TAGAT-PF"];
end

results = struct();
metricRows = table();

fprintf("\n=== Running particle-filter baselines ===\n");

for k = 1:numel(baselineNames)
    name = baselineNames(k);
    fprintf("\n[%d/%d] %s\n",k,numel(baselineNames),name);

    switch name
        case "PF-Fixed"
            rel = [];
        case "PF-Adaptive"
            rel = [];
        case "Ontology-Rule-PF"
            rel = ruleReliability(data,cfg);
        case "GAT-PF"
            rel = pred.GAT;
        case "Ontology-RGAT-PF"
            rel = pred.RGAT;
        case "Ontology-TAGAT-PF"
            rel = pred.TAGAT;
    end

    result = runParticleFilter(data,testIdx,name,rel,cfg,runDir);
    key = matlab.lang.makeValidName(name);
    results.(key) = result;

    m = computeMetrics(data,testIdx,result,cfg);
    m.Model = string(name);
    metricRows = [metricRows; struct2table(m)]; %#ok<AGROW>

    modelDir = fullfile(runDir,name);
    if ~exist(modelDir,"dir"), mkdir(modelDir); end
    save(fullfile(modelDir,"result.mat"),"result","-v7.3");
    writetable(result.trajectory,fullfile(modelDir,"trajectory.csv"));
    writetable(result.stepLog,fullfile(modelDir,"step_log.csv"));
end

metricRows = movevars(metricRows,"Model","Before",1);
writetable(metricRows,fullfile(runDir,"metrics.csv"));

plotFinalResults(data,testIdx,results,pred,ontologyGraph,metricRows,cfg,runDir);
save(fullfile(runDir,"all_results.mat"),"results","metricRows","cfg","testIdx","-v7.3");

fprintf("\n=== Summary ===\n");
disp(metricRows);
fprintf("Results saved to:\n%s\n",runDir);

summary = struct("runDir",runDir,"metrics",metricRows);
end
