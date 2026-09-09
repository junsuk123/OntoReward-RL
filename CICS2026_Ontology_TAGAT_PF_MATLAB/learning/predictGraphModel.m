function pred = predictGraphModel(model,pack,G,mode,cfg) %#ok<INUSD>
%PREDICTGRAPHMODEL Predict reliability and log final-window edge attention.

T = size(pack.X,3);
Y = zeros(4,T);
E = numel(G.src);
A = nan(E,T);

temporalMode = upper(string(mode))=="TAGAT";
window = model.window;

for t=1:T
    [x,~] = makeBatchWindows(pack,t,window);
    [yp,aux] = graphNetForward(model.params,x(:,:,:,1),G,temporalMode);
    Y(:,t) = double(extractdata(yp));

    if ~isempty(aux.edgeAlpha)
        aa = double(extractdata(aux.edgeAlpha));
        if numel(aa)==E
            A(:,t)=aa;
        end
    end
end

pred.Y = clamp(Y,0,1);
pred.edgeAttention = A;
pred.mode = string(mode);
end
