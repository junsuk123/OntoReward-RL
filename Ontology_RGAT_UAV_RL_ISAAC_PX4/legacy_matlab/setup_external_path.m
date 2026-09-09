function root = setup_external_path()
%SETUP_EXTERNAL_PATH Prefer external sim.* while reusing the original algorithms.
%   The original workspace supplies the ontology graph, R-GAT, rewards, PPO and
%   evaluation code and is added read-only, behind this workspace's matlab/src
%   so the external overlays win.
%
%   Set ONTOLOGY_RGAT_ORIGINAL to point at it explicitly; otherwise the known
%   layouts are searched in order. It has moved before, and a hard-coded single
%   path turned that into an unexplained failure in every entry point at once.
root = fileparts(fileparts(mfilename('fullpath')));
parent = fileparts(root);
leaf = fullfile('Ontology_RGAT_UAV_RL_MATLAB','Ontology_RGAT_UAV_RL_MATLAB');
candidates = { ...
    getenv('ONTOLOGY_RGAT_ORIGINAL'), ...
    fullfile(parent,leaf), ...
    fullfile(parent,'past',leaf)};
original = '';
for k = 1:numel(candidates)
    c = candidates{k};
    if ~isempty(c) && isfolder(c) && isfolder(fullfile(c,'src'))
        original = c; break;
    end
end
if isempty(original)
    error('setup_external_path:original', ...
        ['Original workspace not found. Looked for %s under %s and %s.\n' ...
        'Set ONTOLOGY_RGAT_ORIGINAL to its location.'], ...
        leaf,parent,fullfile(parent,'past'));
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
