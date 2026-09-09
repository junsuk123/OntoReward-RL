function report = build_urbannav_features(whichScenario,opts)
%BUILD_URBANNAV_FEATURES Rebuild the 12-feature tables from raw public UrbanNav data.
%
%   build_urbannav_features("all")
%   build_urbannav_features("deep")
%   build_urbannav_features("all",rawRoot="D:\UrbanNav\raw")
%   build_urbannav_features("all",featurePreset="robust")
%
% Reads the u-blox F9P NMEA stream and the raw ground-truth trajectory for each
% scenario, synchronizes them on UTC time-of-day, and writes one epoch-level CSV
% per scenario satisfying DATA_CONTRACT.md except for the soft label.
%
% Expected raw layout under rawRoot (cfg.rawDataDir by default):
%   <rawRoot>/medium/gnss/UrbanNav-HK-Medium-Urban-1.ublox.f9p.nmea
%   <rawRoot>/medium/UrbanNav_TST_GT_raw.txt
%   <rawRoot>/harsh/...    <rawRoot>/deep/...
% Run open_urbannav_dataset for the official download page.
%
% ---------------------------------------------------------------------------
% THIS IS NOT A STRICT REPRODUCTION OF THE PRIOR STUDY.
%
% Output goes to data/urbannav_public/, never to the strict data/real/ slot, and
% the column soft_fault_prob is NOT produced: the prior presentation does not
% specify its residual-to-probability mapping, so inventing one would silently
% fabricate the supervision signal the semi-supervised head is trained on. A
% documented surrogate is written as soft_fault_prob_surrogate, deliberately
% under a different name so validate_data_contract still refuses these tables as
% strict inputs. run_all_real therefore still stops at its data gate until a
% genuine soft_fault_prob is supplied.
%
% Feature provenance (all receiver-reported unless noted):
%   numSV            GNS satellites-used (u-blox reports the full multi-GNSS
%                    count here; GGA truncates it). Switch with opts.numSVSource.
%   hDOP, vDOP       GGA / GSA
%   hAcc             sqrt(stdLat^2 + stdLon^2) from GST
%   vAcc             stdAlt from GST
%   gSpeed           VTG km/h (falls back to RMC knots)
%   CNO_mean/std/gap GSV C/N0 over tracked satellites, deduplicated per SV
%   low_elev_ratio   fraction of tracked satellites below opts.lowElevDeg
%   PR_RMS           RMS of GRS pseudorange residuals
%   Fault_SVID_count number of GRS residuals with |r| > opts.faultResidualM,
%                    plus any satellite flagged by GBS
%   hard_label       GT-based 2-D position error >= opts.faultErrorM (3 m)
%
% The two thresholds and the C/N0-gap definition are NOT specified in the prior
% slides. They are explicit options here and are recorded in the sidecar JSON so
% a later comparison can restate them rather than guess.

arguments
    whichScenario (1,1) string = "all"
    % "literal" reads each feature name the most direct way; "robust" uses the
    % outlier- and scale-insensitive variants.
    opts.featurePreset (1,1) string ...
        {mustBeMember(opts.featurePreset,["literal","robust","custom"])} = "literal"
    opts.numSVSource (1,1) string {mustBeMember(opts.numSVSource,["gns","gga"])} = "gns"
    opts.lowElevDeg (1,1) double = 15
    opts.faultResidualM (1,1) double = 10
    opts.cnoGapMode (1,1) string ...
        {mustBeMember(opts.cnoGapMode,["maxmin","p90p10","max_minus_median","top3_minus_bot3"])} = "maxmin"
    opts.prRmsMode (1,1) string ...
        {mustBeMember(opts.prRmsMode,["rms","median_abs","log_rms","trimmed_rms"])} = "rms"
    opts.faultCountMode (1,1) string ...
        {mustBeMember(opts.faultCountMode,["count","ratio"])} = "count"
    opts.faultErrorM (1,1) double = 3
    opts.maxTimeGapS (1,1) double = 0.6
    opts.receiverPattern (1,1) string = "ublox.f9p"
    opts.residualSignal (1,1) string ...
        {mustBeMember(opts.residualSignal,["primary","worst"])} = "primary"
    % Read the raw logs from somewhere other than data/raw. The public archives
    % are hundreds of megabytes, so pointing at an existing copy beats duplicating.
    opts.rawRoot (1,1) string = ""
    opts.outputSubdir (1,1) string = "urbannav_public"
end

if opts.featurePreset == "robust"
    opts.cnoGapMode     = "p90p10";
    opts.lowElevDeg     = 10;
    opts.prRmsMode      = "log_rms";
    opts.faultCountMode = "ratio";
    opts.faultResidualM = 30;
    opts.numSVSource    = "gga";
end

cfg = cics_config("real");
rawRoot = opts.rawRoot;
if strlength(rawRoot) == 0, rawRoot = string(cfg.rawDataDir); end
whichScenario = lower(whichScenario);
if whichScenario == "all"
    scenarios = ["medium","harsh","deep"];
elseif any(whichScenario == ["medium","harsh","deep"])
    scenarios = whichScenario;
else
    error('Scenario must be "all", "medium", "harsh", or "deep".');
end

outDir = fullfile(cfg.root,'data',char(opts.outputSubdir));
if ~isfolder(outDir), mkdir(outDir); end

fprintf('\n=== Rebuild 12-feature tables from raw public UrbanNav data ===\n');
fprintf('Raw root       : %s\n',rawRoot);
fprintf('Receiver filter: *%s*   Feature preset: %s\n',opts.receiverPattern,opts.featurePreset);
fprintf('Output         : %s\n',outDir);
fprintf('NOT a strict reproduction: soft_fault_prob is not reconstructable from the slides.\n');
if ~isfolder(rawRoot)
    fprintf(2,'\nRaw root does not exist. Download the official archives (GNSS + Ground Truth\n');
    fprintf(2,'for Medium, Harsh and Deep Urban) and lay them out as documented in the help\n');
    fprintf(2,'text of this function. Run open_urbannav_dataset for the download page.\n');
    error('build_urbannav_features:noRawRoot','Raw UrbanNav root not found: %s',rawRoot);
end

rows = {};
for s = scenarios
    fprintf('\n[%s]\n',upper(s));
    scenDir = fullfile(rawRoot,char(s));
    nmeaFile = pick_nmea(scenDir,opts.receiverPattern);
    gtFile   = fullfile(scenDir,char(cfg.urbannav.(char(s)).gtName));
    if nmeaFile == "" || ~isfile(gtFile)
        fprintf(2,'  missing NMEA or GT under %s; skipped.\n',scenDir);
        continue
    end
    fprintf('  NMEA: %s\n',nmeaFile);
    fprintf('  GT  : %s\n',gtFile);

    E = parse_nmea_epochs(nmeaFile,residualSignal=opts.residualSignal);
    G = read_urbannav_gt(gtFile);
    fprintf('  parsed %d NMEA epochs, %d GT epochs\n',height(E),height(G));

    T = synchronize_and_build(E,G,s,opts,cfg);
    if isempty(T)
        fprintf(2,'  no overlapping epochs; skipped.\n');
        continue
    end

    f = fullfile(outDir,char(s)+".csv");
    writetable(T,f);
    nf = sum(T.hard_label==1);
    fprintf('  wrote %s : n=%d, fault=%d (%.1f%%)\n',f,height(T),nf,100*nf/height(T));

    d = cfg.urbannav.slideDiagnostics.(char(s));
    fprintf('  slide diagnostic for reference: n=%d, fault=%d\n',d(1),d(2));

    rows(end+1,:) = {char(s),height(T),nf,nf/height(T),d(1),d(2),char(f)}; %#ok<AGROW>
end

if isempty(rows)
    error('build_urbannav_features:noScenarios', ...
        'No scenario could be rebuilt. Check the raw layout under %s.',rawRoot);
end

report = cell2table(rows,'VariableNames', ...
    {'Scenario','Epochs','Faults','FaultRate','SlideEpochs','SlideFaults','File'});
disp(report);

meta = struct('generated',char(datetime('now','Format','yyyy-MM-dd''T''HH:mm:ss')), ...
    'source',"official public UrbanNav RINEX/NMEA + raw GT", ...
    'rawRoot',char(rawRoot), ...
    'strictReproduction',false, ...
    'softLabelReconstructable',false, ...
    'options',opts,'report',rows);
fid = fopen(fullfile(outDir,'reconstruction_manifest.json'),'w');
fwrite(fid,jsonencode(meta,PrettyPrint=true)); fclose(fid);

write_notice(outDir,opts);
fprintf('\nThese tables are NOT drop-in strict inputs for run_all_real. See %s\n', ...
    fullfile(outDir,'PUBLIC_RECONSTRUCTION_NOTICE.md'));
end

% =========================================================================
function f = pick_nmea(scenDir,pattern)
d = dir(fullfile(scenDir,'gnss','**','*.nmea'));
f = "";
for i = 1:numel(d)
    if contains(d(i).name,pattern) && ~contains(d(i).name,"splitter")
        f = string(fullfile(d(i).folder,d(i).name)); return
    end
end
for i = 1:numel(d)      % fall back to the splitter stream if that is all there is
    if contains(d(i).name,pattern)
        f = string(fullfile(d(i).folder,d(i).name)); return
    end
end
end

function T = synchronize_and_build(E,G,scenario,opts,cfg)
% Epoch pairing and the local ENU frame go through align_nmea_to_gt so that any
% feature-definition study uses identical pairing.
[Em,Gm,posErr,enu] = align_nmea_to_gt(E,G,opts.maxTimeGapS);
if isempty(Em), T = table(); return, end

% ---- the twelve features ------------------------------------------------
F = derive_urbannav_features(Em, ...
    numSVSource=opts.numSVSource, ...
    lowElevDeg=opts.lowElevDeg, ...
    faultResidualM=opts.faultResidualM, ...
    cnoGapMode=opts.cnoGapMode, ...
    prRmsMode=opts.prRmsMode, ...
    faultCountMode=opts.faultCountMode);

T = table();
T.timestamp = Gm.utcTime;
for v = F.Properties.VariableNames
    T.(v{1}) = F.(v{1});
end

% ---- supervision / provenance -------------------------------------------
T.hard_label = double(posErr >= opts.faultErrorM);
T.pos_error_2d = posErr;
T.gnss_east = enu.gnssEast;  T.gnss_north = enu.gnssNorth;
T.gt_east = enu.gtEast;      T.gt_north = enu.gtNorth;
T.fix_quality = Em.fixQuality;
T.num_tracked = Em.numTracked;
T.gt_quality = Gm.Q;
T.receiver_id = repmat(string(opts.receiverPattern),height(T),1);
T.scenario = repmat(string(scenario),height(T),1);

% Documented surrogate ONLY - deliberately not named soft_fault_prob.
T.soft_fault_prob_surrogate = surrogate_soft_label(T.PR_RMS,opts);

% Drop epochs where a required feature is missing; validate_data_contract
% rejects NaN/Inf outright, so reporting the loss here beats failing later.
good = true(height(T),1);
for k = cfg.featureNames
    good = good & isfinite(T.(char(k)));
end
nDrop = sum(~good);
if nDrop > 0
    fprintf('  dropped %d/%d epochs with incomplete features\n',nDrop,height(T));
    for k = cfg.featureNames
        m = sum(~isfinite(T.(char(k))));
        if m > 0, fprintf('     %-18s missing in %d epochs\n',k,m); end
    end
end
T = T(good,:);
end

function p = surrogate_soft_label(prRms,opts)
% A transparent logistic squashing of the residual RMS. This is NOT the prior
% study's estimator; it exists so the code path can be exercised, and it is
% named accordingly so it can never be mistaken for the real soft label.
x = prRms./max(opts.faultResidualM,eps);
p = 1./(1+exp(-(x-1)));
end

function write_notice(outDir,opts)
f = fullfile(outDir,'PUBLIC_RECONSTRUCTION_NOTICE.md');
fid = fopen(f,'w');
fprintf(fid,'# Public UrbanNav reconstruction - NOT a strict reproduction\n\n');
fprintf(fid,'These tables were rebuilt from the official public UrbanNav u-blox F9P NMEA\n');
fprintf(fid,'stream and raw ground truth by `build_urbannav_features.m`.\n\n');
fprintf(fid,'## Why these are not drop-in strict inputs\n\n');
fprintf(fid,'`soft_fault_prob` is absent. The prior presentation does not define its\n');
fprintf(fid,'residual-to-probability estimator, so it cannot be reconstructed. The column\n');
fprintf(fid,'`soft_fault_prob_surrogate` is a documented logistic function of `PR_RMS`\n');
fprintf(fid,'provided only so the training path can be exercised. Renaming it to\n');
fprintf(fid,'`soft_fault_prob` would make `validate_data_contract` accept fabricated\n');
fprintf(fid,'supervision, which is exactly what the strict gate exists to prevent.\n\n');
fprintf(fid,'`run_all_real` reads `data/real/` and will keep refusing to start until a\n');
fprintf(fid,'genuine soft label is available.\n\n');
fprintf(fid,'## Choices not specified by the prior study\n\n');
fprintf(fid,'| Item | Value used | Note |\n|---|---|---|\n');
fprintf(fid,'| feature preset | `%s` | see below |\n',opts.featurePreset);
fprintf(fid,'| numSV source | `%s` | GGA truncates the multi-GNSS count |\n',opts.numSVSource);
fprintf(fid,'| low_elev_ratio threshold | %g deg | not stated in the slides |\n',opts.lowElevDeg);
fprintf(fid,'| Fault_SVID_count residual threshold | %g m | RAIM threshold not stated |\n',opts.faultResidualM);
fprintf(fid,'| CNO_gap definition | `%s` | not stated in the slides |\n',opts.cnoGapMode);
fprintf(fid,'| PR_RMS definition | `%s` | not stated in the slides |\n',opts.prRmsMode);
fprintf(fid,'| Fault_SVID_count mode | `%s` | a raw count scales with tracked SV number |\n',opts.faultCountMode);
fprintf(fid,'| multi-signal residual reduction | `%s` | one residual per satellite |\n',opts.residualSignal);
fprintf(fid,'| hard-label error threshold | %g m | stated in the slides |\n',opts.faultErrorM);
fprintf(fid,'| GT/NMEA max time gap | %g s | 1 Hz alignment tolerance |\n',opts.maxTimeGapS);
fprintf(fid,'\nChanging any of these changes the dataset. Restate them in any write-up.\n');
fprintf(fid,'\n## Feature presets\n\n');
fprintf(fid,'`featurePreset="literal"` (default) reads each feature name the most direct way.\n');
fprintf(fid,'`featurePreset="robust"` replaces the outlier- and scale-sensitive definitions:\n');
fprintf(fid,'`CNO_gap` p90-p10 instead of max-min, `PR_RMS` log-RMS, `Fault_SVID_count` as a\n');
fprintf(fid,'ratio rather than a raw count, low-elevation threshold 10 deg, residual threshold\n');
fprintf(fid,'30 m, numSV from GGA.\n');
fclose(fid);
end
