function [X,Yh,Ys] = concat_windows(A,B,cfg,horizon)
[XA,ha,sa]=make_windows(A,cfg,horizon);
[XB,hb,sb]=make_windows(B,cfg,horizon);
X=cat(3,XA,XB); Yh=[ha hb]; Ys=[sa sb];
end
