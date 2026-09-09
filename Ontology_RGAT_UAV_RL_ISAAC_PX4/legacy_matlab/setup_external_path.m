function root = setup_external_path()
%SETUP_EXTERNAL_PATH Prefer external sim.* while reusing the original algorithms.
root = fileparts(fileparts(mfilename('fullpath')));
original = fullfile(fileparts(root),'Ontology_RGAT_UAV_RL_MATLAB', ...
    'Ontology_RGAT_UAV_RL_MATLAB');
if ~isfolder(original)
    error('Original workspace not found: %s',original);
end
addpath(fullfile(original,'config'),'-end');
addpath(fullfile(original,'src'),'-end');
addpath(fullfile(original,'validation'),'-end');
addpath(fullfile(original,'tests'),'-end');
addpath(fullfile(root,'matlab','src'),'-begin');
addpath(fullfile(root,'matlab'),'-begin');
fprintf('External sim adapter: %s\n',fullfile(root,'matlab','src'));
fprintf('Read-only algorithm source: %s\n',original);
end

