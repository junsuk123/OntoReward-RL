classdef RGATMonitor < handle
    %RGATMONITOR Live view of R-GAT potential training: losses and validation fit.
    properties
        Fig; Cfg; LTrain; LVal; SPred;
    end
    methods
        function obj=RGATMonitor(cfg)
            obj.Cfg=cfg;
            obj.Fig=figure('Name','R-GAT potential training','NumberTitle','off', ...
                'Position',[100 100 960 420]);
            tiledlayout(obj.Fig,1,2,'TileSpacing','compact','Padding','compact');
            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.LTrain=plot(ax,nan,nan,'LineWidth',1.6);
            obj.LVal=plot(ax,nan,nan,'LineWidth',1.6);
            xlabel(ax,'Epoch'); ylabel(ax,'MSE'); title(ax,'Potential regression loss');
            legend(ax,{'train','validation'},'Location','best','Box','off');
            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.SPred=scatter(ax,nan,nan,10,'filled','MarkerFaceAlpha',0.35);
            plot(ax,[-1 1],[-1 1],'k--','LineWidth',1);
            xlabel(ax,'target potential'); ylabel(ax,'predicted potential');
            title(ax,'Validation fit'); xlim(ax,[-1.05 1.05]);
        end
        function update(obj,H,ep,yTrue,yPred)
            x=1:ep;
            set(obj.LTrain,'XData',x,'YData',H.trainLoss(1:ep));
            set(obj.LVal,'XData',x,'YData',H.valLoss(1:ep));
            if nargin>=5 && ~isempty(yPred)
                set(obj.SPred,'XData',yTrue(:),'YData',yPred(:));
            end
            if mod(ep,max(1,obj.Cfg.viz.liveEvery))==0 || ep==numel(H.trainLoss)
                obj.snapshot(H,ep);
            end
        end
        function snapshot(obj,H,ep)
            T=table((1:ep)',H.trainLoss(1:ep),H.valLoss(1:ep), ...
                'VariableNames',{'Epoch','TrainMSE','ValMSE'});
            viz.saveLive(obj.Fig,'rgat_training',obj.Cfg,T);
        end
    end
end
