function M = compute_metrics(y,p,thr)
%COMPUTE_METRICS Classification + calibration metrics without Statistics Toolbox.
y=double(y(:)); p=double(p(:)); pred=p>=thr;
TP=sum(pred==1 & y==1); TN=sum(pred==0 & y==0); FP=sum(pred==1 & y==0); FN=sum(pred==0 & y==1);
precision=TP/max(1,TP+FP); recall=TP/max(1,TP+FN); f1=2*precision*recall/max(eps,precision+recall);
acc=(TP+TN)/numel(y); spec=TN/max(1,TN+FP);
[auroc,auprc]=auc_scores(y,p);
brier=mean((p-y).^2); nll=-mean(y.*log(p+eps)+(1-y).*log(1-p+eps)); ece=ece_score(y,p,10);
M=struct('TP',TP,'TN',TN,'FP',FP,'FN',FN,'Accuracy',acc,'Precision',precision, ...
    'Recall',recall,'Specificity',spec,'F1',f1,'AUROC',auroc,'AUPRC',auprc, ...
    'Brier',brier,'NLL',nll,'ECE',ece);
end

function [rocAuc,prAuc]=auc_scores(y,p)
[pSorted,ord]=sort(p,'descend'); y=y(ord); %#ok<ASGLU>
P=sum(y==1); N=sum(y==0);
tp=cumsum(y==1); fp=cumsum(y==0);
tpr=[0;tp/max(1,P)]; fpr=[0;fp/max(1,N)]; rocAuc=trapz(fpr,tpr);
prec=tp./max(1,tp+fp); rec=tp/max(1,P);
prAuc=trapz([0;rec],[1;prec]);
end

function e=ece_score(y,p,nBins)
e=0; edges=linspace(0,1,nBins+1);
for b=1:nBins
    if b<nBins, m=p>=edges(b)&p<edges(b+1); else, m=p>=edges(b)&p<=edges(b+1); end
    if any(m), e=e+mean(m)*abs(mean(p(m))-mean(y(m))); end
end
end
