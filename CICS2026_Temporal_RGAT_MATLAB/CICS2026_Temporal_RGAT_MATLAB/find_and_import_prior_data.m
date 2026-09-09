function report=find_and_import_prior_data(searchRoot)
%FIND_AND_IMPORT_PRIOR_DATA Find exact processed tables and copy only unambiguous matches.
cfg=cics_config("real");
if nargin<1 || strlength(string(searchRoot))==0, searchRoot=fileparts(cfg.root); end
req=cellstr(cfg.featureNames); scenarios=["medium","harsh","deep"];
keys.medium={'medium','middle','tst'}; keys.harsh={'harsh','mongkok','mk'}; keys.deep={'deep','whampoa','wp'};
files=[dir(fullfile(searchRoot,'**','*.csv'));dir(fullfile(searchRoot,'**','*.txt'))];
candidates=struct('medium',{{}},'harsh',{{}},'deep',{{}}); rows={};
for k=1:numel(files)
    f=fullfile(files(k).folder,files(k).name);
    if contains(lower(f),lower(fullfile(cfg.root,'data'))), continue; end
    try, T=readtable(f,'VariableNamingRule','preserve'); catch, continue; end
    names=T.Properties.VariableNames; nFeat=sum(ismember(req,names));
    hasHard=ismember('hard_label',names)||ismember('pos_error_2d',names); hasSoft=ismember('soft_fault_prob',names);
    sc=""; nm=lower(files(k).name);
    for s=scenarios
        if any(cellfun(@(q)contains(nm,q),keys.(char(s)))), sc=s; break; end
    end
    if nFeat>=4 || hasSoft, rows(end+1,:)={f,nFeat,hasHard,hasSoft,char(sc)}; end %#ok<AGROW>
    if nFeat==numel(req)&&hasHard&&hasSoft&&strlength(sc)>0, candidates.(char(sc)){end+1}=f; end
end
if isempty(rows), disp('No candidate processed tables found.'); else, disp(cell2table(rows,'VariableNames',{'File','MatchedFeatures','HasHard','HasSoft','Scenario'})); end
for s=scenarios
    c=candidates.(char(s));
    if numel(c)==1
        copyfile(c{1},cfg.files.(char(s)),'f'); fprintf('Imported %s -> %s\n',c{1},cfg.files.(char(s)));
    elseif numel(c)>1
        fprintf('%s ambiguous: %d exact matches. Not copied.\n',s,numel(c));
    end
end
report=struct('candidates',candidates,'ready',all(arrayfun(@(s)isfile(cfg.files.(char(s))),scenarios)));
end
