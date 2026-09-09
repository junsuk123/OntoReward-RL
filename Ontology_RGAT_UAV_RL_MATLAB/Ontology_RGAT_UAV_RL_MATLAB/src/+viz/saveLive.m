function saveLive(fig,name,cfg,tbl)
%SAVELIVE Export a live training figure (and its history table) to results/live.
% Lets progress be inspected from outside MATLAB while a headless or background
% run is still going, which a live figure alone cannot do.
if ~isfield(cfg,'viz') || ~isfield(cfg.viz,'liveExport') || ~cfg.viz.liveExport
    return;
end
d=cfg.paths.live; if ~exist(d,'dir'), mkdir(d); end
try
    exportgraphics(fig,fullfile(d,[name '.png']),'Resolution',140);
catch ME
    warning('viz:saveLive','snapshot failed for %s: %s',name,ME.message);
end
if nargin>=4 && ~isempty(tbl)
    try
        writetable(tbl,fullfile(d,[name '.csv']));
    catch ME
        warning('viz:saveLive','history csv failed for %s: %s',name,ME.message);
    end
end
end
