function [model,history] = trainGraphModel(pack,G,mode,trainIdx,valIdx,cfg)
%TRAINGRAPHMODEL Train GAT / R-GAT / TA-GAT reliability predictor.

model = initGraphModel(pack,G,mode,cfg);
p = model.params;
window = model.window;
temporalMode = upper(string(mode))=="TAGAT";

trainIdx = trainIdx(trainIdx>=1);
trainIdx = trainIdx(1:cfg.learning.trainStride:end);
valIdx = valIdx(valIdx>=1);

m=[]; v=[];
iter=0;
bestVal=Inf;
bestP=p;
badEpochs=0;

epochCol=[]; trainLossCol=[]; valLossCol=[];

for epoch=1:cfg.learning.epochs
    order = trainIdx(randperm(numel(trainIdx)));
    losses = [];

    for s=1:cfg.learning.batchSize:numel(order)
        batchIdx = order(s:min(numel(order),s+cfg.learning.batchSize-1));
        [bX,bY] = makeBatchWindows(pack,batchIdx,window);

        [loss,gr] = dlfeval(@modelGradients,p,bX,bY,G,temporalMode,cfg.learning.weightDecay);
        iter=iter+1;
        [p,m,v] = adamUpdateStruct(p,gr,m,v,iter,cfg.learning.learningRate,cfg.learning.gradientClip);
        losses(end+1)=double(extractdata(loss)); %#ok<AGROW>
    end

    tr = mean(losses);
    va = evaluateModelLoss(p,pack,G,valIdx,window,temporalMode);

    epochCol(end+1,1)=epoch; %#ok<AGROW>
    trainLossCol(end+1,1)=tr; %#ok<AGROW>
    valLossCol(end+1,1)=va; %#ok<AGROW>

    fprintf("epoch %02d | train %.5f | val %.5f\n",epoch,tr,va);

    if va < bestVal - 1e-5
        bestVal=va; bestP=p; badEpochs=0;
    else
        badEpochs=badEpochs+1;
        if badEpochs>=cfg.learning.patience
            fprintf("early stop at epoch %d\n",epoch);
            break;
        end
    end
end

model.params = bestP;
history = table(epochCol,trainLossCol,valLossCol, ...
    'VariableNames',["Epoch","TrainLoss","ValLoss"]);
end

function loss = evaluateModelLoss(p,pack,G,idx,window,temporalMode)
if isempty(idx), loss=NaN; return; end
idx = idx(1:max(1,ceil(numel(idx)/180)):end); % bounded validation cost
acc=0;
for k=1:numel(idx)
    [x,y] = makeBatchWindows(pack,idx(k),window);
    yp = graphNetForward(p,x(:,:,:,1),G,temporalMode);
    yy = double(extractdata(yp));
    acc = acc + mean((yy-double(y)).^2);
end
loss = acc/numel(idx);
end
