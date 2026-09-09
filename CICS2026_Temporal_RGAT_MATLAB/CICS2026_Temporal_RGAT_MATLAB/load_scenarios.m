function D = load_scenarios(cfg)
%LOAD_SCENARIOS Load and validate medium/harsh/deep tables.
for s=["medium","harsh","deep"]
    f=cfg.files.(char(s));
    if ~isfile(f), error('Missing data file: %s',f); end
    T=readtable(f,'VariableNamingRule','preserve');
    T=validate_data_contract(T,cfg,s);
    S=struct();
    S.X=single(T{:,cellstr(cfg.featureNames)});
    S.hard=single(T.hard_label(:));
    S.soft=single(T.soft_fault_prob(:));
    S.table=T;
    D.(char(s))=S;
end
end
