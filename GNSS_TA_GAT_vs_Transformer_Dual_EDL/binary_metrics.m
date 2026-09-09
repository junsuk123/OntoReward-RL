function m=binary_metrics(y,pred)
y=logical(y(:)); pred=logical(pred(:));
TP=sum(pred & y);
TN=sum(~pred & ~y);
FP=sum(pred & ~y);
FN=sum(~pred & y);

m.TP=TP; m.TN=TN; m.FP=FP; m.FN=FN;
m.precision=TP/max(TP+FP,1);
m.recall=TP/max(TP+FN,1);
m.f1=2*m.precision*m.recall/max(m.precision+m.recall,eps);
m.accuracy=(TP+TN)/max(numel(y),1);
m.fpr=FP/max(FP+TN,1);
end
