function T = validate_data_contract(T,cfg,scenario)
%VALIDATE_DATA_CONTRACT Strictly validate input columns and labels.
names=string(T.Properties.VariableNames);
missing=cfg.featureNames(~ismember(cfg.featureNames,names));
if ~isempty(missing)
    error('%s missing feature columns: %s',scenario,strjoin(missing,', '));
end
if ~ismember("hard_label",names)
    if ismember("pos_error_2d",names)
        T.hard_label=double(T.pos_error_2d>=cfg.hardFaultThresholdM);
    else
        error('%s requires hard_label or pos_error_2d.',scenario);
    end
end
if ~ismember("soft_fault_prob",names)
    if cfg.allowSurrogateSoftLabel && ismember("soft_fault_prob_surrogate",names)
        % Non-strict mode only. The surrogate is used in memory and never written
        % back under the strict column name, so the table on disk stays honest.
        warning('cics:surrogateSoftLabel', ...
            ['%s: using soft_fault_prob_surrogate as the weak label. This is NOT the ' ...
             'prior study''s residual-to-probability estimator, so any Dual-EDL result ' ...
             'from this run is not comparable with the reported baseline.'],scenario);
        T.soft_fault_prob=weak_label_surrogate(T,cfg,scenario);
    elseif cfg.mode=="real"
        error(['%s has no soft_fault_prob. Strict prior-study reproduction is not permitted ' ...
            'without the original residual-based soft label.'],scenario);
    else
        error('%s table missing soft_fault_prob.',scenario);
    end
end
if any(~ismember(unique(T.hard_label),[0 1]))
    error('%s hard_label must be binary 0/1.',scenario);
end
if any(T.soft_fault_prob<0 | T.soft_fault_prob>1 | ~isfinite(T.soft_fault_prob))
    error('%s soft_fault_prob must be finite in [0,1].',scenario);
end
for f=cfg.featureNames
    v=T.(char(f));
    if any(~isfinite(v)), error('%s contains NaN/Inf in %s.',scenario,f); end
end
end
