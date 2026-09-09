function phi = predict(P,graph)
%PREDICT Potential of one graph, on the numeric path.
%   Called twice per control step by reward.proposedPBRS, inside a loop that is
%   paced to PX4's 50 Hz. The dlarray path costs about 11 ms per call because
%   every one of the ~26 edges per layer is a separate traced operation, which
%   put two evaluations at 28 ms against a 20 ms budget. Unwrapping the
%   parameters once and running the same batched arithmetic in plain doubles
%   costs about 0.12 ms and needs no autodiff tape, because nothing here is
%   differentiated.
Pn=rgat.unwrap(P);
X=graph.X;
if isa(X,'dlarray'), X=double(extractdata(X)); end
phi=double(rgat.forward(Pn,X,graph));
end
