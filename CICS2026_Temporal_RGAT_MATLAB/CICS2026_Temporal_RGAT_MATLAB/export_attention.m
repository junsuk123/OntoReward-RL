function export_attention(model,Xwindow,cfg)
%EXPORT_ATTENTION Save node/relation/time/head/layer attention summaries.
layers=model.net.Layers;
idx=[];
for k=1:numel(layers)
    if isa(layers(k),'TemporalRGATLayer')
        idx=k;
        break
    end
end
if isempty(idx), return; end
layer=layers(idx); S=layer.analyze(single(Xwindow));
out=cfg.outputDir;
writetable(table(cfg.featureNames',S.node,'VariableNames',{'Node','Importance'}),fullfile(out,'attention_node.csv'));
writetable(table(layer.RelationNames',S.relation,'VariableNames',{'Relation','Importance'}),fullfile(out,'attention_relation.csv'));
writetable(table((0:layer.MaxLag)',S.timeLag,'VariableNames',{'Lag','Importance'}),fullfile(out,'attention_time_lag.csv'));
writetable(table((1:layer.NumHeads)',S.head,'VariableNames',{'Head','Importance'}),fullfile(out,'attention_head.csv'));
writetable(table((1:layer.NumLayers)',S.layer,'VariableNames',{'Layer','Importance'}),fullfile(out,'attention_layer.csv'));

figure('Name','Temporal R-GAT Attention Summary','Color','w');
tiledlayout(2,2);
nexttile; bar(S.node); xticks(1:numel(cfg.featureNames)); xticklabels(cfg.featureNames); xtickangle(45); title('Node');
nexttile; bar(S.relation); xticks(1:numel(layer.RelationNames)); xticklabels(layer.RelationNames); xtickangle(35); title('Relation');
nexttile; bar(0:layer.MaxLag,S.timeLag); xlabel('Lag'); title('Time-lag');
nexttile; bar(S.layer); xlabel('Layer'); title('Layer gating');
exportgraphics(gcf,fullfile(out,'attention_summary.png'),'Resolution',180);
end
