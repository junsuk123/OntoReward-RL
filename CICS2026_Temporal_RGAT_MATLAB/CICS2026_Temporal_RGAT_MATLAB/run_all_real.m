function run_all_real(mode)
%RUN_ALL_REAL Real-data experiment on UrbanNav-HK.
%
%   run_all_real()          strict if data/real is complete, otherwise falls back
%                           to the labelled non-strict run on the public
%                           reconstruction in data/urbannav_public.
%   run_all_real("strict")  strict only; refuses to start without the prior
%                           study's own soft label. Use this for anything that
%                           will be compared against the reported baseline.
%
% The fallback exists because raw UrbanNav yields 13 of the 14 required columns:
% build_urbannav_features can rebuild the 12 features and hard_label, but the
% prior presentation never defines its residual-to-probability mapping, so
% soft_fault_prob cannot be reconstructed. The fallback trains the weak head on a
% documented surrogate instead, and every artefact it writes says so.
if nargin<1, mode="auto"; end
mode=string(mode);
root=fileparts(mfilename('fullpath'));
addpath(root); addpath(fullfile(root,'tests'));
check_environment();

cfg=cics_config("real");
missing=string.empty;
for s=["medium","harsh","deep"]
    if ~isfile(cfg.files.(char(s))), missing(end+1)=string(cfg.files.(char(s))); end %#ok<AGROW>
end
if isempty(missing)
    run_benchmark("real");
    return
end

fprintf('\nMissing strict processed files:\n'); disp(missing');
recon=cics_config("nonstrict");
haveRecon=all(arrayfun(@(s)isfile(recon.files.(char(s))),["medium","harsh","deep"]));

if mode=="strict" || ~haveRecon
    fprintf('Options, in decreasing order of scientific strength:\n');
    fprintf('  1. Place the prior study''s own processed tables (12 features + hard_label\n');
    fprintf('     + soft_fault_prob) in %s.\n',cfg.dataDir);
    fprintf('  2. find_and_import_prior_data(<searchRoot>) to locate and copy them.\n');
    fprintf('  3. build_urbannav_features("all") rebuilds the 12 features and hard_label\n');
    fprintf('     from the raw public UrbanNav logs into data/urbannav_public. That covers\n');
    fprintf('     13 of the 14 required columns, but NOT soft_fault_prob: the prior\n');
    fprintf('     presentation never defines its residual-to-probability mapping.\n');
    if haveRecon
        fprintf('     Those tables are present. run_all_real() without "strict" will run on\n');
        fprintf('     them using a documented surrogate weak label.\n');
    end
    fprintf('  For raw UrbanNav download links run: open_urbannav_dataset\n');
    error('Strict real-data benchmark cannot start without all three processed tables.');
end

banner();
run_benchmark("nonstrict");
banner();
end

function banner()
line=repmat('*',1,78);
fprintf('\n%s\n',line);
fprintf('*** NON-STRICT RUN ON THE PUBLIC UrbanNav RECONSTRUCTION\n');
fprintf('*** Features and hard_label are real, rebuilt from the official raw logs.\n');
fprintf('*** The weak label is soft_fault_prob_surrogate, a logistic function of\n');
fprintf('*** PR_RMS - NOT the prior study''s residual-to-probability estimator.\n');
fprintf('*** Therefore: Dual-EDL numbers here are NOT comparable with the reported\n');
fprintf('*** F1=0.950 baseline, and the reproduction gate is meaningless in this mode.\n');
fprintf('*** Usable for architecture and ablation comparison. Not a reproduction.\n');
fprintf('%s\n',line);
end
