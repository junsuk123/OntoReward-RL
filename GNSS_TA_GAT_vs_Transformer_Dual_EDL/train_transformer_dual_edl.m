function model = train_transformer_dual_edl(X,yHard,ySoft,cfg)
[C,~,B] = size(X);
D = cfg.transformer.dModel;
H = cfg.transformer.numHeads;
F = cfg.transformer.ffnDim;
assert(mod(D,H)==0,'dModel must be divisible by numHeads');

params = init_baseline_params(C,D,F,cfg.transformer.numLayers);
avg = cell(size(params)); avgSq = cell(size(params));
for i=1:numel(params), avg{i}=[]; avgSq{i}=[]; end

order = 1:B;
iter = 0;

for epoch=1:cfg.training.epochs
    order = order(randperm(B));
    epLoss = 0; nBatch=0;
    for s=1:cfg.training.batchSize:B
        iter=iter+1; nBatch=nBatch+1;
        id = order(s:min(s+cfg.training.batchSize-1,B));

        xb = dlarray(single(X(:,:,id)));
        yh = single(yHard(id));
        ys = single(ySoft(id));

        [loss,grads] = dlfeval(@baseline_gradients,params,xb,yh,ys,cfg,epoch);

        grads = clip_grads(grads,cfg.training.gradClip);
        for i=1:numel(params)
            [params{i},avg{i},avgSq{i}] = adamupdate( ...
                params{i},grads{i},avg{i},avgSq{i},iter,cfg.training.learnRate);
        end
        epLoss = epLoss + double(gather(extractdata(loss)));

        if mod(iter,cfg.training.verboseEvery)==0
            fprintf('Baseline epoch %d/%d iter %d loss %.4f\n', ...
                epoch,cfg.training.epochs,iter,epLoss/nBatch);
        end
    end
    fprintf('Baseline epoch %d mean loss %.4f\n',epoch,epLoss/max(nBatch,1));
end

model.params=params;
model.type='Transformer-Dual EDL';
end

function [loss,grads] = baseline_gradients(params,X,yh,ys,cfg,epoch)
[p1,u1,p2,u2,~,~,a1,a2] = baseline_forward(params,X,cfg);
loss = dual_edl_loss(a1,a2,yh,ys,cfg,epoch);
grads = cell(size(params));
[grads{:}] = dlgradient(loss,params{:});
end

function params = init_baseline_params(C,D,F,L)
params={};
params{end+1}=winit(D,C); params{end+1}=zinit(D,1); % embed
for l=1:L
    params{end+1}=winit(D,D); % Wq
    params{end+1}=winit(D,D); % Wk
    params{end+1}=winit(D,D); % Wv
    params{end+1}=winit(D,D); % Wo
    params{end+1}=winit(F,D); params{end+1}=zinit(F,1);
    params{end+1}=winit(D,F); params{end+1}=zinit(D,1);
end
params{end+1}=winit(2,D); params{end+1}=zinit(2,1); % head1
params{end+1}=winit(2,D); params{end+1}=zinit(2,1); % head2
end

function x=winit(r,c)
x=dlarray(single(randn(r,c)*sqrt(2/max(c,1))));
end
function x=zinit(r,c)
x=dlarray(zeros(r,c,'single'));
end
