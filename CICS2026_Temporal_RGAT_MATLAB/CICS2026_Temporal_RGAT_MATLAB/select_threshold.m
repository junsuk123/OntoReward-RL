function [thr,f1] = select_threshold(y,p,fallback)
%SELECT_THRESHOLD Pick the decision threshold that maximises F1 on held-out data.
%
% A fixed 0.5 threshold assumes the evaluation set carries the same class prior
% the model was fitted under. It does not here: faults cluster in time, so a
% chronological split gives train 0.589, val 0.331 and test 0.421 fault rates.
% A model that ranks well (AUROC 0.84) was therefore scoring F1 0.497 - exactly
% the all-positive degenerate value - purely because every probability landed
% above 0.5. The threshold is chosen on VALIDATION only and then applied
% unchanged to test, so this stays a legitimate held-out estimate.
if nargin<3, fallback=0.5; end
y=double(y(:)); p=double(p(:));
if isempty(y)||all(y==y(1)), thr=fallback; f1=NaN; return; end
cand=unique([0.05:0.01:0.95, reshape(p,1,[])]);
cand=cand(cand>0 & cand<1);
bestF=-Inf; thr=fallback;
for t=cand
    pred=p>=t;
    TP=sum(pred & y==1); FP=sum(pred & y==0); FN=sum(~pred & y==1);
    f=2*TP/max(1,2*TP+FP+FN);
    if f>bestF, bestF=f; thr=t; end
end
f1=bestF;
end
