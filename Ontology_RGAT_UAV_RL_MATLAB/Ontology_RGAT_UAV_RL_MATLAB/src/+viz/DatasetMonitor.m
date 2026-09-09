classdef DatasetMonitor < handle
    %DATASETMONITOR Live view of expert rollout collection for the R-GAT dataset.
    % Surfaces the expert success rate as it is collected: a flat zero here means
    % the potential target has no positive examples to learn from.
    properties
        Fig; Cfg; LRate; LCum; LSamp; SNoise;
    end
    methods
        function obj=DatasetMonitor(cfg)
            obj.Cfg=cfg;
            obj.Fig=figure('Name','R-GAT dataset generation','NumberTitle','off', ...
                'Position',[120 120 1180 380]);
            tiledlayout(obj.Fig,1,3,'TileSpacing','compact','Padding','compact');
            ax=nexttile; hold(ax,'on'); grid(ax,'on'); ylim(ax,[0 1]);
            obj.LRate=plot(ax,nan,nan,'LineWidth',1.8,'Color',[0.05 0.55 0.30]);
            xlabel(ax,'Episode'); ylabel(ax,'Success rate');
            title(ax,'Expert moving success rate');
            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.LCum=plot(ax,nan,nan,'LineWidth',1.8,'Color',[0.10 0.35 0.75]);
            obj.LSamp=plot(ax,nan,nan,'LineWidth',1.2,'Color',[0.85 0.40 0.10]);
            xlabel(ax,'Episode'); ylabel(ax,'Count');
            legend(ax,{'successful episodes','samples/episode'},'Location','best','Box','off');
            title(ax,'Collection progress');
            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.SNoise=scatter(ax,nan,nan,24,'filled','MarkerFaceAlpha',0.6);
            xlabel(ax,'action noise sigma'); ylabel(ax,'success'); ylim(ax,[-0.1 1.1]);
            title(ax,'Outcome vs perturbation');
        end
        function update(obj,S,ep)
            x=1:ep; win=max(1,min(20,round(ep/4)));
            set(obj.LRate,'XData',x,'YData',movmean(S.success(1:ep),win));
            set(obj.LCum,'XData',x,'YData',cumsum(S.success(1:ep)));
            set(obj.LSamp,'XData',x,'YData',S.samples(1:ep));
            set(obj.SNoise,'XData',S.noise(1:ep),'YData',S.success(1:ep));
            if mod(ep,max(1,obj.Cfg.viz.liveEvery))==0 || ep==numel(S.success)
                obj.snapshot(S,ep);
            end
        end
        function snapshot(obj,S,ep)
            T=table((1:ep)',S.success(1:ep),S.noise(1:ep),S.samples(1:ep), ...
                'VariableNames',{'Episode','Success','NoiseStd','Samples'});
            viz.saveLive(obj.Fig,'rgat_dataset',obj.Cfg,T);
        end
    end
end
