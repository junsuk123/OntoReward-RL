function D = load_gnss_csv(file,cfg,envName)
if ~isfile(file)
    error(['Missing file: %s\n' ...
        'Place medium.csv, harsh.csv, deep.csv under data/.'],file);
end

T = readtable(file,'VariableNamingRule','preserve');
vn = string(T.Properties.VariableNames);

% Small alias normalization
alias = containers.Map( ...
    {'meanCN0','stdCN0','fault_SVID_count','2D_error','posError2D', ...
     'CNO_mean','CNO_std','CNO_gap', ...
     'hard_label'}, ...
    {'CN0_mean','CN0_std','Fault_SVID_count','pos_error_2d','pos_error_2d', ...
     'CN0_mean','CN0_std','CN0_gap', ...
     'fault_label'});

% The dataset ships soft_fault_prob_surrogate, but it is saturated near 1.0
% for ~68% of rows while the true fault rate is 31-68%, and correlates only
% ~0.25 with pos_error_2d. Off by default -> the GT-error sigmoid below is
% used instead. Set cfg.useSurrogateSoftLabel = true to train on it.
if isfield(cfg,'useSurrogateSoftLabel') && cfg.useSurrogateSoftLabel
    alias('soft_fault_prob_surrogate') = 'soft_fault_prob';
end

ak = keys(alias);
for k = 1:numel(ak)
    old = ak{k};
    new = alias(old);
    if any(vn == old) && ~any(vn == new)
        T.Properties.VariableNames{find(vn==old,1)} = new;
        vn = string(T.Properties.VariableNames);
    end
end

for i = 1:numel(cfg.features)
    if ~ismember(cfg.features{i},T.Properties.VariableNames)
        error('Required feature column missing in %s: %s',file,cfg.features{i});
    end
end

X = single(T{:,cfg.features});

if any(~isfinite(X),'all')
    fprintf('[%s] Non-finite feature values detected -> linear fill + nearest endpoints\n',envName);
    X = fillmissing(X,'linear',1,'EndValues','nearest');
end

if ismember('fault_label',T.Properties.VariableNames)
    yHard = single(T.fault_label(:) > 0);
elseif ismember('pos_error_2d',T.Properties.VariableNames)
    err = single(T.pos_error_2d(:));
    yHard = single(err > cfg.faultThresholdM);
else
    error('%s needs fault_label or pos_error_2d.',file);
end

if ismember('soft_fault_prob',T.Properties.VariableNames)
    ySoft = single(T.soft_fault_prob(:));
elseif ismember('pos_error_2d',T.Properties.VariableNames)
    err = single(T.pos_error_2d(:));
    z = (err - cfg.faultThresholdM) / cfg.softLabelTemperature;
    ySoft = single(1 ./ (1 + exp(-z)));
    fprintf('[%s] soft_fault_prob absent -> GT-error sigmoid fallback used.\n',envName);
else
    ySoft = yHard;
    fprintf('[%s] soft label absent -> hard label reused as fallback.\n',envName);
end

ySoft = min(max(ySoft,0),1);

D.X = X;                % [T x C]
D.yHard = yHard;        % [T x 1]
D.ySoft = ySoft;        % [T x 1]
D.env = envName;
D.n = size(X,1);
end
