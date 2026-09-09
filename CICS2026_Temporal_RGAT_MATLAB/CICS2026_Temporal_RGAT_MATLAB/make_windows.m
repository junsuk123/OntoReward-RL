function [X,Yhard,Ysoft,meta] = make_windows(S,cfg,horizon)
%MAKE_WINDOWS Create causal windows ending at t and target at t+horizon.
if nargin<3, horizon=0; end
W=cfg.window; n=size(S.X,1); F=size(S.X,2);
last=n-horizon; nWin=last-W+1;
if nWin<=0, error('Not enough samples for W=%d horizon=%d.',W,horizon); end
X=zeros(F,W,nWin,'single');
Yhard=zeros(1,nWin,'single'); Ysoft=zeros(1,nWin,'single');
endEpoch=zeros(nWin,1); targetEpoch=zeros(nWin,1);
for k=1:nWin
    e=k+W-1; target=e+horizon;
    X(:,:,k)=S.X(k:e,:).';
    Yhard(k)=S.hard(target); Ysoft(k)=S.soft(target);
    endEpoch(k)=e; targetEpoch(k)=target;
end
meta=table((1:nWin)',endEpoch,targetEpoch,'VariableNames',{'window_id','end_epoch','target_epoch'});
end
