function model = train_tagat_dual_edl(Stats,A,yHard,ySoft,cfg)
[S,N,B]=size(Stats); %#ok<ASGLU>
D=cfg.tagat.hiddenDim;

params=init_tagat_params(S,D,cfg.tagat.numLayers);
avg=cell(size(params)); avgSq=cell(size(params));
for i=1:numel(params), avg{i}=[]; avgSq{i}=[]; end

order=1:B; iter=0;
for epoch=1:cfg.training.epochs
    order=order(randperm(B));
    epLoss=0; nBatch=0;
    for s=1:cfg.training.batchSize:B
        iter=iter+1; nBatch=nBatch+1;
        id=order(s:min(s+cfg.training.batchSize-1,B));

        st=dlarray(single(Stats(:,:,id)));
        ab=single(A(:,:,id));
        yh=single(yHard(id));
        ys=single(ySoft(id));

        [loss,grads]=dlfeval(@tagat_gradients,params,st,ab,yh,ys,cfg,epoch);
        grads=clip_grads(grads,cfg.training.gradClip);

        for i=1:numel(params)
            [params{i},avg{i},avgSq{i}]=adamupdate( ...
                params{i},grads{i},avg{i},avgSq{i},iter,cfg.training.learnRate);
        end
        epLoss=epLoss+double(gather(extractdata(loss)));

        if mod(iter,cfg.training.verboseEvery)==0
            fprintf('TA-GAT epoch %d/%d iter %d loss %.4f\n', ...
                epoch,cfg.training.epochs,iter,epLoss/nBatch);
        end
    end
    fprintf('TA-GAT epoch %d mean loss %.4f\n',epoch,epLoss/max(nBatch,1));
end

model.params=params;
model.type='Ontology-Guided TA-GAT + Dual EDL';
end

function [loss,grads]=tagat_gradients(params,Stats,A,yh,ys,cfg,epoch)
[~,~,~,~,~,~,a1,a2]=tagat_forward(params,Stats,A,cfg);
loss=dual_edl_loss(a1,a2,yh,ys,cfg,epoch);
grads=cell(size(params));
[grads{:}]=dlgradient(loss,params{:});
end

function params=init_tagat_params(S,D,L)
params={};
params{end+1}=winit(D,S); params{end+1}=zinit(D,1);
for l=1:L
    params{end+1}=winit(D,D);   % W
    params{end+1}=winit(D,1);   % a_src
    params{end+1}=winit(D,1);   % a_dst
end
params{end+1}=winit(2,D); params{end+1}=zinit(2,1);
params{end+1}=winit(2,D); params{end+1}=zinit(2,1);
end

function x=winit(r,c)
x=dlarray(single(randn(r,c)*sqrt(2/max(c,1))));
end
function x=zinit(r,c)
x=dlarray(zeros(r,c,'single'));
end
