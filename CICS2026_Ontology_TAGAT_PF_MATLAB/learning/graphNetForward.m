function [y,aux] = graphNetForward(p,Xseq,G,temporalMode)
%GRAPHNETFORWARD Custom GAT / R-GAT / temporal-adaptive R-GAT forward pass.
%
% Relation-aware message:
%   m_ij = W_common*x_j + W_rel(r_ij)*x_j + node_embedding_j
% Attention additionally uses relation embedding.
%
% TA mode:
%   - processes a sliding window recurrently
%   - edge score includes transformed absolute feature change
%   - previous node hidden state is gated into current state

if ~isa(Xseq,"dlarray")
    Xseq = dlarray(single(Xseq));
end

W = size(Xseq,3);
D = size(p.Wcommon,1);
N = G.numNodes;

Hprev = dlarray(single(zeros(D,N)));
Xprev = Xseq(:,:,1);
lastAlpha = [];

temporalFlag = single(logical(temporalMode));
gamma = 1./(1+exp(-p.gammaRaw));

for tt=1:W
    X = Xseq(:,:,tt);
    Hbase = p.Wcommon*X + p.nodeEmb;

    nodeCols = cell(1,N);
    alphaParts = cell(1,N);

    for i=1:N
        eidx = find(G.dst==i);
        if isempty(eidx)
            nodeCols{i} = tanh(Hbase(:,i) + temporalFlag*gamma*Hprev(:,i));
            alphaParts{i} = [];
            continue;
        end

        scores = cell(numel(eidx),1);
        msgs = cell(numel(eidx),1);

        for kk=1:numel(eidx)
            e = eidx(kk);
            j = G.src(e);
            r = G.rel(e);

            msg = Hbase(:,j) + p.Wrel(:,:,r)*X(:,j);
            score = sum(p.aSrc.*msg) + sum(p.aDst.*Hbase(:,i)) + ...
                    sum(p.aRel.*p.relEmb(:,r));

            dX = abs(X(:,j)-Xprev(:,j));
            tFeat = p.Wtemp*dX;
            score = score + temporalFlag*sum(p.aTemp.*tFeat);

            score = max(score,0) + 0.2*min(score,0); % leaky ReLU
            scores{kk} = score;
            msgs{kk} = msg;
        end

        a = stableSoftmax(cat(1,scores{:}));
        agg = msgs{1}.*a(1);
        for kk=2:numel(msgs)
            agg = agg + msgs{kk}.*a(kk);
        end

        recurrent = temporalFlag*gamma*Hprev(:,i);
        nodeCols{i} = tanh(agg + Hbase(:,i) + recurrent);
        alphaParts{i} = a;
    end

    H = cat(2,nodeCols{:});
    Hprev = H;
    Xprev = X;
    lastAlpha = catNonEmpty(alphaParts);
end

pooled = reshape(H(:,G.outputNodes),[],1);
logits = p.Wout*pooled + p.bout;
y = 1./(1+exp(-logits));

aux.edgeAlpha = lastAlpha;
aux.hidden = H;
end

function a = stableSoftmax(s)
% Softmax over an unformatted dlarray column. The toolbox softmax requires a
% DataFormat argument that does not apply to these edge-score vectors.
s = s - max(s,[],1);
e = exp(s);
a = e ./ sum(e,1);
end

function out = catNonEmpty(parts)
valid = ~cellfun(@isempty,parts);
if any(valid)
    out = cat(1,parts{valid});
else
    out = [];
end
end
