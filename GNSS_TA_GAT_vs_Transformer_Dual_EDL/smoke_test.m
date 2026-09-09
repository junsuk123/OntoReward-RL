clear; clc;
cfg=default_config();
rng(cfg.seed);

C=numel(cfg.features);
T=600;

% Synthetic trajectories only for shape/runtime checking.
% Do NOT report these values as research results.
makeSeq=@(name,shift) struct( ...
    'X',single(randn(T,C)+shift), ...
    'yHard',single(rand(T,1)>0.75), ...
    'ySoft',single(rand(T,1)), ...
    'env',name, ...
    'n',T);

seq={makeSeq('Middle',0),makeSeq('Harsh',0.4)};
[mu,sigma]=fit_standardizer(seq);
seq=apply_standardizer(seq,mu,sigma);
[X,yh,ys]=build_windows(seq,cfg);

% Keep smoke test short.
cfg.training.epochs=1;
cfg.training.batchSize=16;
cfg.training.verboseEvery=10;
X=X(:,:,1:min(96,size(X,3)));
yh=yh(1:size(X,3)); ys=ys(1:size(X,3));

fprintf('X size = [%s]\n',num2str(size(X)));

b=train_transformer_dual_edl(X,yh,ys,cfg);
[pb,ub]=predict_transformer_dual_edl(b,X,cfg);
fprintf('Baseline output size: p=%s u=%s\n',mat2str(size(pb)),mat2str(size(ub)));

Aont=build_ontology_adjacency(cfg.features);
[S,A]=build_temporal_graph_inputs(X,Aont,cfg);
g=train_tagat_dual_edl(S,A,yh,ys,cfg);
[pg,ug]=predict_tagat_dual_edl(g,S,A,cfg);
fprintf('TA-GAT output size: p=%s u=%s\n',mat2str(size(pg)),mat2str(size(ug)));

fprintf('\nSmoke test completed. Synthetic results are NOT research results.\n');
