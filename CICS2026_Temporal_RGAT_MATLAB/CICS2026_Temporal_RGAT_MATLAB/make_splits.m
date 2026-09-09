function SP = make_splits(D,cfg,horizon)
%MAKE_SPLITS Build train/validation/test window sets under a stated protocol.
%
% cfg.split.mode:
%   "pooled_chronological"  each scenario is cut chronologically into
%                           train / val / test. Every environment is represented
%                           in all three sets, and no future sample ever informs
%                           a past one.
%   "cross_scenario"        the original protocol: train on medium+harsh, test
%                           on deep. Retained because it is a legitimate
%                           question (does this transfer to an unseen
%                           environment?) but on this dataset the answer is no -
%                           see SPLIT_PROTOCOL.md.
%
% PURGE. Windows are built with stride 1 over a length-W history, so two windows
% whose end epochs are less than W apart share raw samples. Cutting a
% chronological split at a single index therefore leaks up to W-1 shared samples
% across the boundary. cfg.split.purgeWindows windows are dropped on each side of
% every boundary so that no test window shares a single epoch with a train window.
if nargin<3, horizon=0; end
scen=["medium","harsh","deep"];
mode=string(cfg.split.mode);

Xtr=[];Xva=[];Xte=[];
yhTr=[];yhVa=[];yhTe=[]; ysTr=[];ysVa=[];ysTe=[];
metaTe=table();

switch mode
    case "cross_scenario"
        for s=["medium","harsh"]
            [X,yh,ys]=make_windows(D.(char(s)),cfg,horizon);
            n=numel(yh); k=max(1,floor(n*(1-cfg.split.valFrac)));
            g=cfg.split.purgeWindows;
            Xtr=cat(3,Xtr,X(:,:,1:k)); yhTr=[yhTr yh(1:k)]; ysTr=[ysTr ys(1:k)];
            v=min(n,k+g)+1;
            if v<=n
                Xva=cat(3,Xva,X(:,:,v:n)); yhVa=[yhVa yh(v:n)]; ysVa=[ysVa ys(v:n)];
            end
        end
        [Xte,yhTe,ysTe,metaTe]=make_windows(D.deep,cfg,horizon);
        metaTe.scenario=repmat("deep",height(metaTe),1);

    case "pooled_chronological"
        f1=cfg.split.trainFrac; f2=cfg.split.trainFrac+cfg.split.valFrac;
        g=cfg.split.purgeWindows;
        for s=scen
            [X,yh,ys,mt]=make_windows(D.(char(s)),cfg,horizon);
            n=numel(yh);
            iTrEnd=floor(n*f1); iVaEnd=floor(n*f2);
            tr=1:iTrEnd;
            va=(iTrEnd+g+1):iVaEnd;
            te=(iVaEnd+g+1):n;
            if isempty(va)||isempty(te)
                error('Scenario %s too short for the requested split and purge.',s);
            end
            Xtr=cat(3,Xtr,X(:,:,tr)); yhTr=[yhTr yh(tr)]; ysTr=[ysTr ys(tr)];
            Xva=cat(3,Xva,X(:,:,va)); yhVa=[yhVa yh(va)]; ysVa=[ysVa ys(va)];
            Xte=cat(3,Xte,X(:,:,te)); yhTe=[yhTe yh(te)]; ysTe=[ysTe ys(te)];
            m=mt(te,:); m.scenario=repmat(s,height(m),1); metaTe=[metaTe;m]; %#ok<AGROW>
        end
    otherwise
        error('Unknown split mode %s',mode);
end

SP=struct('Xtrain',Xtr,'YhTrain',yhTr,'YsTrain',ysTr, ...
          'Xval',Xva,'YhVal',yhVa,'YsVal',ysVa, ...
          'Xtest',Xte,'YhTest',yhTe,'YsTest',ysTe, ...
          'metaTest',metaTe,'mode',mode);
end
