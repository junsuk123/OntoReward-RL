function model = build_model(name,cfg)
%BUILD_MODEL Build one benchmark model.
name=string(name); G=build_ontology_graph(cfg,false);
switch name
    case "Transformer"
        net=build_transformer_network(12,2,cfg,[]); lossMode="softmax_hard";
    case "TransEDL-Hard"
        net=build_transformer_network(12,2,cfg,[]); lossMode="edl_hard";
    case "TransEDL-Soft"
        net=build_transformer_network(12,2,cfg,[]); lossMode="edl_soft";
    case "Transformer-DualEDL"
        net=build_transformer_network(12,4,cfg,[]); lossMode="dual_edl";
    case "R-GAT-DualEDL"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,cfg.rgat.numHeads,cfg.rgat.numLayers,0, ...
            cfg.rgat.timeDim,false,"rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    case "R-GAT+Transformer-DualEDL"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,cfg.rgat.numHeads,cfg.rgat.numLayers,0, ...
            cfg.rgat.timeDim,false,"rgat");
        net=build_transformer_network(12,4,cfg,rg); lossMode="dual_edl";
    case "TemporalR-GAT-DualEDL"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,cfg.rgat.numHeads,cfg.rgat.numLayers,cfg.rgat.maxLag, ...
            cfg.rgat.timeDim,cfg.rgat.useTimeEncoding,"temporal_rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    case "TemporalR-GAT-NoTimeEncoding"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,cfg.rgat.numHeads,cfg.rgat.numLayers,cfg.rgat.maxLag, ...
            cfg.rgat.timeDim,false,"temporal_rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    case "TemporalR-GAT-NoRelationType"
        G2=build_ontology_graph(cfg,true);
        rg=TemporalRGATLayer(cfg.rgat.hidden,G2,cfg.rgat.numHeads,cfg.rgat.numLayers,cfg.rgat.maxLag, ...
            cfg.rgat.timeDim,true,"temporal_rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    case "TemporalR-GAT-1Head"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,1,cfg.rgat.numLayers,cfg.rgat.maxLag, ...
            cfg.rgat.timeDim,true,"temporal_rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    case "TemporalR-GAT-1Layer"
        rg=TemporalRGATLayer(cfg.rgat.hidden,G,cfg.rgat.numHeads,1,cfg.rgat.maxLag, ...
            cfg.rgat.timeDim,true,"temporal_rgat");
        net=rgat_head(rg);
        lossMode="dual_edl";
    otherwise
        error('Unknown model: %s',name);
end
model=struct('name',name,'net',net,'lossMode',lossMode);
end

function net = rgat_head(rg)
%RGAT_HEAD Graph-attention trunk -> normalisation -> temporal pooling -> logits.
%
% The layer normalisation is load-bearing, not cosmetic. TemporalRGATLayer ends
% in a mean over the 12 feature nodes, and globalAveragePooling1d then takes a
% mean over the 30 time steps. Two un-normalised means shrink the
% input-dependent part of the representation by ~46x: measured at
% initialisation, the pooled features varied by sd 0.012 across windows against
% the Transformer's 0.53, so every window produced almost the same logits
% (fault probability spanned only [0.4984, 0.5023]) and the model sat at
% single-class output for its whole run. The Transformer never hit this because
% each of its encoder blocks ends in a layerNormalizationLayer that restores
% unit scale. Normalising here gives the graph trunk the same footing: pooled sd
% across windows becomes 0.56.
net=dlnetwork([ ...
    sequenceInputLayer(12,Normalization="none",Name="input"); ...
    rg; ...
    layerNormalizationLayer(Name="rgat_norm"); ...
    globalAveragePooling1dLayer(Name="pool"); ...
    fullyConnectedLayer(4,Name="output")]);
end
