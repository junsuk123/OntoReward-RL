function [trainIdx,valIdx,testIdx] = splitIndices(T,cfg)
nTrain=max(1,floor(cfg.split.trainFraction*T));
nVal=max(1,floor(cfg.split.valFraction*T));
nVal=min(nVal,T-nTrain-1);
trainIdx=1:nTrain;
valIdx=nTrain+(1:nVal);
testIdx=(nTrain+nVal+1):T;
end
