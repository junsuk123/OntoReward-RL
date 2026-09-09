function run_unit_tests()
%RUN_UNIT_TESTS Data-contract, model-build, forward-shape, EDL and attention tests.
cfg=cics_config("demo");
D=load_scenarios(cfg); [D,~,~]=normalize_scenarios(D,cfg);
[X,yh,ys]=concat_windows(D.medium,D.harsh,cfg,0);
assert(size(X,1)==12 && size(X,2)==cfg.window);
assert(numel(yh)==size(X,3) && all(ys>=0&ys<=1));
G=build_ontology_graph(cfg,false);
assert(isequal(size(G.adjacency),[12 12 6]));

% EDL decoder sanity.
a=single([2 3;4 1]);
[p,u]=edl_decode(a,"edl_hard");
assert(all(p>=0&p<=1) && all(u>0&u<=1));

% Temporal R-GAT forward and analyzer sanity.
m=build_model("TemporalR-GAT-DualEDL",cfg);
xb=permute(X(:,:,1:min(2,size(X,3))),[1 3 2]); % C x B x T
raw=predict(m.net,dlarray(single(xb),'CBT'));
r=extractdata(raw);
assert(size(r,1)==4 && size(r,2)==size(xb,2));
[p2,u2]=edl_decode(single(r),m.lossMode);
assert(all(isfinite(p2)) && all(p2>=0&p2<=1));
assert(all(isfinite(u2)) && all(u2>0&u2<=1));

layers=m.net.Layers; rg=[];
for k=1:numel(layers)
    if isa(layers(k),'TemporalRGATLayer'), rg=layers(k); break; end
end
assert(~isempty(rg));
S=rg.analyze(single(X(:,:,1)));
assert(abs(sum(S.node)-1)<1e-4);
assert(abs(sum(S.relation)-1)<1e-4);
assert(abs(sum(S.timeLag)-1)<1e-4);
assert(abs(sum(S.head)-1)<1e-4);
assert(abs(sum(S.layer)-1)<1e-4);

% Prior Transformer forward path sanity.
tm=build_model("Transformer-DualEDL",cfg);
rawT=predict(tm.net,dlarray(single(xb),'CBT'));
rT=extractdata(rawT);
assert(size(rT,1)==4 && size(rT,2)==size(xb,2));

% ---- Regression tests for the 2026-08-19 training fixes --------------------

% 1. Masked EDL KL must vanish when all evidence sits on the correct class.
%    The unmasked form penalised correct evidence too, so CE+KL was minimised
%    at zero evidence and every EDL head collapsed to p=0.5.
alpha=single([1;21]); Tone=single([0;1]);
assert(abs(dirichlet_kl_uniform(Tone+(1-Tone).*alpha))<1e-4, ...
    'masked Dirichlet KL must be ~0 when evidence is entirely on the true class');
assert(dirichlet_kl_uniform(alpha)>1.5, ...
    'unmasked KL is large here - that is precisely why it must be masked');

% 2. The R-GAT trunk must be normalised: its logits have to move when the input
%    moves. Un-normalised, window-to-window sd was ~0.012 against 0.53 for the
%    Transformer, so every window produced the same prediction.
assert(any(strcmp({m.net.Layers.Name},'rgat_norm')),'rgat_norm layer missing');
nw=min(64,size(X,3));
xs=permute(X(:,:,1:nw),[1 3 2]);
rgRaw=double(extractdata(predict(m.net,dlarray(single(xs),'CBT'))));
assert(mean(std(rgRaw,0,2))>0.05, ...
    'R-GAT logits barely vary across windows - trunk normalisation is missing');

% 3. select_threshold must beat a fixed 0.5 when the class prior shifts.
yv=[zeros(1,80) ones(1,20)];
pv=0.6+0.3*yv+0.02*sin(1:100);   % well ranked, but everything sits above 0.5
[tsel,fsel]=select_threshold(yv,pv,0.5);
m05=compute_metrics(yv,pv,0.5);
assert(tsel>0.5 && fsel>m05.F1,'threshold selection failed to improve on 0.5');

% 4. make_splits must purge window overlap so that no evaluation window shares
%    a raw epoch with a training window.
cfgS=cfg; cfgS.split.mode="pooled_chronological";
SPs=make_splits(D,cfgS,0);
assert(~isempty(SPs.Xval) && ~isempty(SPs.Xtest),'splits must be non-empty');
[~,~,~,mt]=make_windows(D.medium,cfgS,0);
nM=height(mt); iTr=floor(nM*cfgS.split.trainFrac);
iVa=iTr+cfgS.split.purgeWindows+1;
assert(mt.end_epoch(iVa)-mt.end_epoch(iTr)>=cfgS.window, ...
    'purge gap too small - train and eval windows still share raw samples');

fprintf('All unit/forward-shape/training-fix tests passed.\n');
end
