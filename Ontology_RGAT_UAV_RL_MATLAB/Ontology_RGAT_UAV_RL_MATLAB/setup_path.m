function root = setup_path()
%SETUP_PATH Add project folders to the MATLAB path.
root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root,'config'));
addpath(fullfile(root,'src'));
addpath(fullfile(root,'validation'));
addpath(fullfile(root,'tests'));
if ~exist(fullfile(root,'results'),'dir'), mkdir(fullfile(root,'results')); end
fprintf('[setup] Project root: %s\n', root);
end
