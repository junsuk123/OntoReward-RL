classdef TrainingMonitor < handle
    %TRAININGMONITOR Live PPO dashboard: return, outcome mix, losses, exploration.
    % The terminal-outcome panel is the one that matters most here: a policy that
    % learns to hover out the clock shows up as a rising 'timeout' fraction with a
    % flat success rate, which a return curve alone hides.
    properties
        Fig; Name; Cfg; StatusNames;
        LRet; LRetMA; LSucc; LLen; LLenMA; LActor; LCritic; LStd; LStatus;
    end
    methods
        function obj=TrainingMonitor(figTitle,name,cfg)
            obj.Name=name; obj.Cfg=cfg;
            obj.StatusNames={'success','unsafe_touchdown','flight_failure','timeout'};
            obj.Fig=figure('Name',figTitle,'NumberTitle','off','Position',[60 60 1240 720]);
            tiledlayout(obj.Fig,2,3,'TileSpacing','compact','Padding','compact');

            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.LRet=plot(ax,nan,nan,'-','Color',[0.72 0.79 0.90],'LineWidth',0.8);
            obj.LRetMA=plot(ax,nan,nan,'-','Color',[0.10 0.35 0.75],'LineWidth',1.8);
            xlabel(ax,'Episode'); ylabel(ax,'Return'); title(ax,'Episode return');
            legend(ax,{'raw','moving mean'},'Location','best','Box','off');

            ax=nexttile; hold(ax,'on'); grid(ax,'on'); ylim(ax,[0 1]);
            obj.LSucc=plot(ax,nan,nan,'-','Color',[0.05 0.55 0.30],'LineWidth',1.8);
            xlabel(ax,'Episode'); ylabel(ax,'Success rate'); title(ax,'Moving success rate');

            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.LLen=plot(ax,nan,nan,'-','Color',[0.95 0.84 0.72],'LineWidth',0.8);
            obj.LLenMA=plot(ax,nan,nan,'-','Color',[0.85 0.40 0.10],'LineWidth',1.8);
            xlabel(ax,'Episode'); ylabel(ax,'Steps'); title(ax,'Episode length');

            ax=nexttile; grid(ax,'on');
            yyaxis(ax,'left');  obj.LActor=plot(ax,nan,nan,'LineWidth',1.5); ylabel(ax,'Actor loss');
            yyaxis(ax,'right'); obj.LCritic=plot(ax,nan,nan,'LineWidth',1.5); ylabel(ax,'Critic loss');
            xlabel(ax,'Episode'); title(ax,'PPO losses');

            ax=nexttile; hold(ax,'on'); grid(ax,'on');
            obj.LStd=plot(ax,nan,nan,'-','Color',[0.45 0.20 0.65],'LineWidth',1.8);
            xlabel(ax,'Episode'); ylabel(ax,'exp(logStd)'); title(ax,'Exploration std');

            ax=nexttile; hold(ax,'on'); grid(ax,'on'); ylim(ax,[0 1]);
            cols=[0.05 0.55 0.30; 0.85 0.33 0.10; 0.64 0.08 0.18; 0.35 0.35 0.35];
            for i=1:numel(obj.StatusNames)
                obj.LStatus(i)=plot(ax,nan,nan,'LineWidth',1.5,'Color',cols(i,:));
            end
            xlabel(ax,'Episode'); ylabel(ax,'Fraction'); title(ax,'Terminal outcome mix');
            legend(ax,strrep(obj.StatusNames,'_','\_'),'Location','best','Box','off');
        end

        function update(obj,H,k)
            x=(1:k)'; win=max(1,min(50,round(k/5)));
            set(obj.LRet,'XData',x,'YData',H.return(1:k));
            set(obj.LRetMA,'XData',x,'YData',movmean(H.return(1:k),win));
            set(obj.LSucc,'XData',x,'YData',movmean(H.success(1:k),win));
            set(obj.LLen,'XData',x,'YData',H.length(1:k));
            set(obj.LLenMA,'XData',x,'YData',movmean(H.length(1:k),win));
            set(obj.LActor,'XData',x,'YData',H.actorLoss(1:k));
            set(obj.LCritic,'XData',x,'YData',H.criticLoss(1:k));
            set(obj.LStd,'XData',x,'YData',H.logStdExp(1:k));
            s=H.status(1:k);
            for i=1:numel(obj.StatusNames)
                set(obj.LStatus(i),'XData',x,'YData',movmean(double(s==obj.StatusNames{i}),win));
            end
            if mod(k,max(1,obj.Cfg.viz.liveEvery))==0, obj.snapshot(H,k); end
        end

        function snapshot(obj,H,k)
            T=table((1:k)',H.iteration(1:k),H.return(1:k),H.success(1:k),H.length(1:k), ...
                H.actorLoss(1:k),H.criticLoss(1:k),H.logStdExp(1:k),H.status(1:k), ...
                'VariableNames',{'Episode','Iteration','Return','Success','Steps', ...
                                 'ActorLoss','CriticLoss','PolicyStd','Status'});
            viz.saveLive(obj.Fig,sprintf('ppo_%s',obj.Name),obj.Cfg,T);
        end
    end
end
