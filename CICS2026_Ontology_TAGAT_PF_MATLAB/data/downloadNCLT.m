function paths = downloadNCLT(cfg)
%DOWNLOADNCLT Download selected NCLT raw files from official UMich hosting.
%
% Downloads are verified by extracting them. A transfer can complete with the
% expected byte count and still be corrupt, so a failed extraction deletes the
% archive and re-downloads once before giving up.

session = char(cfg.data.session);
rawRoot = fullfile(cfg.paths.raw,session);
if ~exist(rawRoot,"dir"), mkdir(rawRoot); end

base = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu";

paths.sensorArchive = fullfile(rawRoot,session + "_sen.tar.gz");
paths.hokuyoArchive = fullfile(rawRoot,session + "_hokuyo.tar.gz");
paths.groundTruth = fullfile(rawRoot,"groundtruth_" + session + ".csv");
paths.extractRoot = fullfile(rawRoot,"extracted");

urls.sensor = base + "/sensor_data/" + session + "_sen.tar.gz";
urls.hokuyo = base + "/hokuyo_data/" + session + "_hokuyo.tar.gz";
urls.gt = base + "/ground_truth/groundtruth_" + session + ".csv";

fetchIfMissing(paths.sensorArchive,urls.sensor,"NCLT sensor archive",cfg);
fetchIfMissing(paths.groundTruth,urls.gt,"NCLT ground truth",cfg);
if cfg.data.downloadHokuyo
    fetchIfMissing(paths.hokuyoArchive,urls.hokuyo,"NCLT Hokuyo archive",cfg);
end

if ~exist(paths.extractRoot,"dir"), mkdir(paths.extractRoot); end

extractVerified(paths.sensorArchive,urls.sensor,paths.extractRoot, ...
    fullfile(paths.extractRoot,".sensor_done"),"sensor","odometry_mu_100hz.csv",cfg);

if cfg.data.downloadHokuyo
    extractVerified(paths.hokuyoArchive,urls.hokuyo,paths.extractRoot, ...
        fullfile(paths.extractRoot,".hokuyo_done"),"Hokuyo","hokuyo_30m.bin",cfg);
end
end

function fetchIfMissing(destPath,url,label,cfg)
if exist(destPath,"file")
    return;
end
if ~cfg.data.autoDownload
    error("Missing %s: %s",label,destPath);
end
fprintf("Downloading %s...\n%s\n",label,url);
websave(destPath,url);
end

function extractVerified(archivePath,url,extractRoot,markerPath,label,expectedFile,cfg)
%EXTRACTVERIFIED Extract once, verifying the archive actually yields its payload.

if exist(markerPath,"file") && strlength(findFileRecursive(extractRoot,expectedFile))>0
    return;
end

maxAttempts = 2;
for attempt = 1:maxAttempts
    fprintf("Extracting %s archive...\n",label);
    ok = true;
    try
        untar(archivePath,extractRoot);
    catch ME
        ok = false;
        reason = ME.message;
    end

    if ok && strlength(findFileRecursive(extractRoot,expectedFile))==0
        ok = false;
        reason = sprintf("archive did not contain %s",expectedFile);
    end

    if ok
        fclose(fopen(markerPath,"w"));
        return;
    end

    fprintf(2,"%s archive is unusable (%s).\n",label,reason);
    if attempt == maxAttempts || ~cfg.data.autoDownload
        error(["Could not extract the NCLT %s archive after %d attempt(s).\n" ...
               "Delete %s and retry, or download it manually from:\n%s"], ...
               label,attempt,archivePath,url);
    end

    fprintf("Re-downloading %s archive (previous copy was corrupt)...\n",label);
    delete(archivePath);
    websave(archivePath,url);
end
end
