classdef TemporalRGATLayer < nnet.layer.Layer & nnet.layer.Formattable
    %TEMPORALRGATLAYER Ontology-constrained relation-time graph attention.
    % Input: N x B x T (CBT), N=12 feature nodes.
    % Output: D x B x T (CBT), graph-pooled relational-temporal state.
    % Attention domain is expanded from (node,relation) to
    % (node,relation,time-lag). Only causal lags delta>=0 are used.

    properties
        Adjacency
        RelationNames
        NumNodes
        NumRelations
        HiddenSize
        NumHeads
        NumLayers
        MaxLag
        TimeDim
        UseTimeEncoding
    end
    properties (Learnable)
        InputScale       % D x N
        NodeEmbedding    % D x N
        WQ               % D x D x R x H x L
        WK
        WV
        RelationBias     % R x H x L
        TimeWeight       % P x H x L
        HeadGate         % D x H x L
        HeadBias         % H x L
        LayerGate        % D x L
        LayerBias        % L x 1
    end

    methods
        function layer=TemporalRGATLayer(hiddenSize,G,numHeads,numLayers,maxLag,timeDim,useTimeEncoding,name)
            layer.Name=name;
            layer.Description="Ontology-constrained Temporal Relational Graph Attention";
            layer.Adjacency=single(G.adjacency);
            layer.RelationNames=G.relations;
            layer.NumNodes=size(G.adjacency,1);
            layer.NumRelations=size(G.adjacency,3);
            layer.HiddenSize=hiddenSize;
            layer.NumHeads=numHeads;
            layer.NumLayers=numLayers;
            layer.MaxLag=maxLag;
            layer.TimeDim=timeDim;
            layer.UseTimeEncoding=logical(useTimeEncoding);
            D=hiddenSize; N=layer.NumNodes; R=layer.NumRelations; H=numHeads; L=numLayers;
            s=sqrt(2/(D+D));
            layer.InputScale=single(0.08*randn(D,N));
            layer.NodeEmbedding=single(0.02*randn(D,N));
            layer.WQ=single(s*randn(D,D,R,H,L));
            layer.WK=single(s*randn(D,D,R,H,L));
            layer.WV=single(s*randn(D,D,R,H,L));
            layer.RelationBias=single(zeros(R,H,L));
            layer.TimeWeight=single(0.02*randn(timeDim,H,L));
            layer.HeadGate=single(0.02*randn(D,H,L));
            layer.HeadBias=single(zeros(H,L));
            layer.LayerGate=single(0.02*randn(D,L));
            layer.LayerBias=single(zeros(L,1));
        end

        function Y=predict(layer,X), Y=doForward(layer,X); end
        function Y=forward(layer,X), Y=doForward(layer,X); end

        function S=analyze(layer,X)
            % X is numeric N x T for one window. Returns interpretable summaries.
            X=single(X);
            if size(X,1)~=layer.NumNodes, error('Expected %d nodes.',layer.NumNodes); end
            [~,~,S]=numericForward(layer,X);
        end
    end

    methods (Access=private)
        function Y=doForward(layer,X)
            % Vectorised over batch and time. The previous implementation looped
            % over every (batch,time,head,lag,relation) tuple with 24x24 matmuls,
            % which made real-mode training (50 epochs, 30-step windows, lag 5)
            % computationally impossible and produced an enormous autodiff tape.
            % Here every (batch,time) column is a page of a batched matmul.
            fmt=dims(X); Xu=stripdims(X);
            N=size(Xu,1); B=size(Xu,2); T=size(Xu,3);
            if N~=layer.NumNodes, error('TemporalRGAT expected %d input features, got %d.',layer.NumNodes,N); end
            D=layer.HiddenSize; R=layer.NumRelations; H=layer.NumHeads; L=layer.NumLayers;
            BT=B*T; maxLag=min(layer.MaxLag,T-1);

            % Node states for every (batch,time) column: D x N x BT with b fastest.
            Hc=layer.InputScale.*reshape(Xu,[1 N BT])+layer.NodeEmbedding;

            % Additive ontology mask. exp() drives masked pairs to exactly zero in
            % single precision, so no separate multiplicative mask is needed.
            pen=cell(R,1);
            for r=1:R, pen{r}=(1-layer.Adjacency(:,:,r))*single(-1e4); end

            outs=cell(L,1);
            for l=1:L
                % Source states per causal lag. Because b is the fastest index the
                % target columns with t>lag form the contiguous block (lag*B+1):BT,
                % so causality needs no masking - only a shift.
                src=cell(maxLag+1,1); nTgt=zeros(maxLag+1,1);
                for dlt=0:maxLag
                    nTgt(dlt+1)=BT-dlt*B;
                    src{dlt+1}=Hc(:,:,1:nTgt(dlt+1));
                end
                Hheads=cell(H,1); headScore=cell(H,1);
                for h=1:H
                    % Sc = Hc' * (WQ'*WK/sqrt(D)) * Hsrc, so the query side is
                    % computed once per relation instead of once per relation-lag.
                    Gq=cell(R,1);
                    for r=1:R
                        Mr=(layer.WQ(:,:,r,h,l).'*layer.WK(:,:,r,h,l))/sqrt(single(D));
                        Gq{r}=pagemtimes(Hc,'transpose',Mr,'none');   % N x D x BT
                    end
                    % Pass 1: scores and the joint (relation,lag) row maximum.
                    Sc=cell(R,maxLag+1); rowMax=[];
                    for dlt=0:maxLag
                        off=BT-nTgt(dlt+1); mxLag=[];
                        tb=timeBias(layer,dlt,h,l);
                        for r=1:R
                            bias=pen{r}+(layer.RelationBias(r,h,l)+tb);
                            s=pagemtimes(Gq{r}(:,:,off+1:BT),'none',src{dlt+1},'none')+bias;
                            Sc{r,dlt+1}=s; mx=max(s,[],2);
                            if isempty(mxLag), mxLag=mx; else, mxLag=max(mxLag,mx); end
                        end
                        mxLag=pad_lead(mxLag,off,N,1,single(-1e9));
                        if isempty(rowMax), rowMax=mxLag; else, rowMax=max(rowMax,mxLag); end
                    end
                    % Pass 2: joint softmax denominator over (source,relation,lag).
                    E=cell(R,maxLag+1); denom=[];
                    for dlt=0:maxLag
                        off=BT-nTgt(dlt+1); dLag=[];
                        for r=1:R
                            e=exp(Sc{r,dlt+1}-rowMax(:,:,off+1:BT));
                            E{r,dlt+1}=e; se=sum(e,2);
                            if isempty(dLag), dLag=se; else, dLag=dLag+se; end
                        end
                        dLag=pad_lead(dLag,off,N,1,single(0));
                        if isempty(denom), denom=dLag; else, denom=denom+dLag; end
                    end
                    % Pass 3: messages. The denominator depends only on the target
                    % node, so it is factored out of the per-edge attention weights
                    % and WV is applied once per relation after the lag sum.
                    msg=[];
                    for r=1:R
                        Pr=[];
                        for dlt=0:maxLag
                            off=BT-nTgt(dlt+1);
                            p=pagemtimes(src{dlt+1},'none',E{r,dlt+1},'transpose');
                            p=pad_lead(p,off,D,N,single(0));
                            if isempty(Pr), Pr=p; else, Pr=Pr+p; end
                        end
                        Pr=reshape(layer.WV(:,:,r,h,l)*reshape(Pr,[D N*BT]),[D N BT]);
                        if isempty(msg), msg=Pr; else, msg=msg+Pr; end
                    end
                    msg=msg./(reshape(denom,[1 N BT])+single(1e-8));
                    hout=tanh(Hc+msg);
                    Hheads{h}=hout;
                    mh=reshape(mean(hout,2),[D BT]);
                    headScore{h}=sum(layer.HeadGate(:,h,l).*mh,1)+layer.HeadBias(h,l);
                end
                gam=softmax_rows(cat(1,headScore{:}));    % H x BT
                comb=[];
                for h=1:H
                    c=reshape(gam(h,:),[1 1 BT]).*Hheads{h};
                    if isempty(comb), comb=c; else, comb=comb+c; end
                end
                Hc=comb; outs{l}=Hc;
            end

            % Layer gating. The graph-pooling mean is linear, so the pooled output
            % is formed straight from the per-layer node means.
            means=cell(L,1); layerScore=cell(L,1);
            for l=1:L
                means{l}=reshape(mean(outs{l},2),[D BT]);
                layerScore{l}=sum(layer.LayerGate(:,l).*means{l},1)+layer.LayerBias(l);
            end
            lam=softmax_rows(cat(1,layerScore{:}));       % L x BT
            Z=[];
            for l=1:L
                c=means{l}.*lam(l,:);
                if isempty(Z), Z=c; else, Z=Z+c; end
            end
            Y=dlarray(reshape(Z,[D B T]),fmt);
        end

        function tb=timeBias(layer,dlt,h,l)
            if ~layer.UseTimeEncoding
                tb=layer.TimeWeight(1,h,l)*0;
            else
                phi=lagEncoding(layer,dlt);
                tb=sum(layer.TimeWeight(:,h,l).*phi);
            end
        end

        function phi=lagEncoding(layer,dlt)
            P=layer.TimeDim; phi=zeros(P,1,'single');
            scale=max(1,layer.MaxLag);
            x=single(dlt/scale);
            k=1;
            while k<=P
                f=single(2^floor((k-1)/2));
                phi(k)=sin(pi*f*x);
                if k+1<=P, phi(k+1)=cos(pi*f*x); end
                k=k+2;
            end
        end

        function [Z,Hfinal,S]=numericForward(layer,X)
            D=layer.HiddenSize; N=layer.NumNodes; T=size(X,2);
            Hseq=zeros(D,N,T,'single');
            for t=1:T
                Hseq(:,:,t)=extractNumeric(layer.InputScale).*repmat(reshape(X(:,t),1,N),D,1)+extractNumeric(layer.NodeEmbedding);
            end
            outs=cell(layer.NumLayers,1);
            nodeImp=zeros(N,1); relImp=zeros(layer.NumRelations,1); lagImp=zeros(layer.MaxLag+1,1);
            headAvg=zeros(layer.NumHeads,1);
            for l=1:layer.NumLayers
                Hnext=zeros(size(Hseq),'single');
                for t=1:T
                    current=Hseq(:,:,t); Hheads=zeros(D,N,layer.NumHeads,'single'); hs=zeros(layer.NumHeads,1,'single');
                    for h=1:layer.NumHeads
                        maxLag=min(layer.MaxLag,t-1); rowMax=-1e9*ones(N,1,'single');
                        entries={}; q=0;
                        for dlt=0:maxLag
                            source=Hseq(:,:,t-dlt); tb=extractNumeric(timeBias(layer,dlt,h,l));
                            for r=1:layer.NumRelations
                                q=q+1; M=layer.Adjacency(:,:,r);
                                Q=extractNumeric(layer.WQ(:,:,r,h,l))*current;
                                K=extractNumeric(layer.WK(:,:,r,h,l))*source;
                                V=extractNumeric(layer.WV(:,:,r,h,l))*source;
                                Sc=(Q.'*K)/sqrt(D)+extractNumeric(layer.RelationBias(r,h,l))+tb;
                                Sc=Sc+(1-M)*single(-1e4);
                                entries{q}=struct('S',Sc,'V',V,'M',M,'r',r,'lag',dlt); %#ok<AGROW>
                                rowMax=max(rowMax,max(Sc,[],2));
                            end
                        end
                        denom=zeros(N,1,'single');
                        for k=1:q
                            entries{k}.E=exp(entries{k}.S-rowMax).*entries{k}.M;
                            denom=denom+sum(entries{k}.E,2);
                        end
                        msg=zeros(D,N,'single');
                        for k=1:q
                            A=entries{k}.E./(denom+1e-8);
                            msg=msg+entries{k}.V*A.';
                            if t==T
                                nodeImp=nodeImp+sum(A,1).';
                                relImp(entries{k}.r)=relImp(entries{k}.r)+sum(A,'all');
                                lagImp(entries{k}.lag+1)=lagImp(entries{k}.lag+1)+sum(A,'all');
                            end
                        end
                        hout=tanh(current+msg); Hheads(:,:,h)=hout;
                        m=mean(hout,2); hs(h)=sum(extractNumeric(layer.HeadGate(:,h,l)).*m)+extractNumeric(layer.HeadBias(h,l));
                    end
                    gamma=softmax_numeric(hs); headAvg=headAvg+gamma/T/layer.NumLayers;
                    comb=zeros(D,N,'single'); for h=1:layer.NumHeads, comb=comb+gamma(h)*Hheads(:,:,h); end
                    Hnext(:,:,t)=comb;
                end
                Hseq=Hnext; outs{l}=Hseq;
            end
            Hfinal=zeros(size(Hseq),'single'); layerAvg=zeros(layer.NumLayers,1,'single');
            for t=1:T
                ls=zeros(layer.NumLayers,1,'single');
                for l=1:layer.NumLayers
                    m=mean(outs{l}(:,:,t),2);
                    ls(l)=sum(extractNumeric(layer.LayerGate(:,l)).*m)+extractNumeric(layer.LayerBias(l));
                end
                lam=softmax_numeric(ls); layerAvg=layerAvg+lam/T;
                tmp=zeros(D,N,'single'); for l=1:layer.NumLayers, tmp=tmp+lam(l)*outs{l}(:,:,t); end
                Hfinal(:,:,t)=tmp;
            end
            Z=squeeze(mean(Hfinal,2));
            S.node=nodeImp/(sum(nodeImp)+eps);
            S.relation=relImp/(sum(relImp)+eps);
            S.timeLag=lagImp/(sum(lagImp)+eps);
            S.head=headAvg/(sum(headAvg)+eps);
            S.layer=layerAvg/(sum(layerAvg)+eps);
        end
    end
end

function y=softmax_numeric(x)
x=x-max(x); e=exp(x); y=e/(sum(e)+eps('single'));
end
function x=extractNumeric(x)
if isa(x,'dlarray'), x=extractdata(x); end
if isa(x,'gpuArray'), x=gather(x); end
x=single(x);
end
function A=pad_lead(A,off,d1,d2,fillValue)
%PAD_LEAD Restore full BT length for a lag whose first OFF columns are invalid.
if off>0
    A=cat(3,repmat(fillValue,[d1 d2 off]),A);
end
end
function Y=softmax_rows(Xs)
%SOFTMAX_ROWS Column-wise softmax over dim 1 with the same epsilon guard
%as the scalar softmax used by the numeric analyzer path.
Xs=Xs-max(Xs,[],1); E=exp(Xs); Y=E./(sum(E,1)+single(1e-8));
end
