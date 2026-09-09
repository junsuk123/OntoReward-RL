function E = parse_nmea_epochs(nmeaFile,opts)
%PARSE_NMEA_EPOCHS Parse a u-blox NMEA log into a per-epoch table.
%
%   E = parse_nmea_epochs(file)
%
% Returns one row per UTC epoch with the raw receiver-reported quantities that
% the prior-study feature set is built from. Nothing is smoothed, filtered or
% invented here; derived features are computed in build_urbannav_features.
%
% Sentences used
%   GGA  fix quality, satellites used, HDOP, altitude, position
%   GNS  multi-constellation satellites used (u-blox reports the full count here,
%        while GGA truncates to the legacy GPS-style field)
%   GSA  PDOP/HDOP/VDOP
%   GSV  per-satellite elevation and C/N0
%   GST  per-axis position standard deviations (-> hAcc / vAcc)
%   GRS  per-satellite range residuals (-> PR_RMS / fault counting)
%   GBS  RAIM fault detection
%   RMC  speed over ground, date
%   VTG  speed over ground (km/h)
%
% Epoch segmentation: a new epoch begins when a time-stamped sentence reports a
% different UTC time-of-day. GSV and GSA carry no timestamp, but u-blox emits
% them after the RMC/GGA of the same epoch, so they attach to the open epoch.

arguments
    nmeaFile (1,1) string
    % How to collapse a satellite that is reported on several signals (L1/L2/..):
    %   "primary" : keep the first GRS sentence per constellation (the primary
    %               signal used for the fix) - unbiased, conventional
    %   "worst"   : keep the largest-magnitude residual across signals - biases
    %               PR_RMS upward, but is more sensitive for fault counting
    opts.residualSignal (1,1) string ...
        {mustBeMember(opts.residualSignal,["primary","worst"])} = "primary"
end

lines = readlines(nmeaFile);
lines = strip(lines);
lines = lines(startsWith(lines,"$"));
% drop the *HH checksum suffix
star = strfind(lines,"*");
for i = 1:numel(lines)
    if ~isempty(star{i})
        lines(i) = extractBefore(lines(i),star{i}(end));
    end
end

n = numel(lines);
epochs = cell(n,1); nE = 0;
cur = [];

for i = 1:n
    f = split(lines(i),",");
    tag = f(1);
    if strlength(tag) < 6, continue, end
    typ = extractAfter(tag,3);
    talker = extractBetween(tag,2,3);

    % ---- epoch segmentation -------------------------------------------------
    t = NaN;
    switch typ
        case {"RMC","GGA","GNS","GST","GRS","GBS"}
            t = tod_seconds(getf(f,2));
        case "GLL"
            t = tod_seconds(getf(f,6));
    end
    if ~isnan(t)
        if isempty(cur) || cur.tod ~= t
            if ~isempty(cur), nE = nE+1; epochs{nE} = cur; end
            cur = new_epoch(t);
        end
    end
    if isempty(cur), continue, end   % untimed sentence before any fix

    % ---- field extraction ---------------------------------------------------
    switch typ
        case "GGA"
            cur.lat      = dm_to_deg(getf(f,3),getf(f,4));
            cur.lon      = dm_to_deg(getf(f,5),getf(f,6));
            cur.fixQual  = num(getf(f,7));
            cur.numSV_gga= num(getf(f,8));
            cur.hdop_gga = num(getf(f,9));
            cur.alt      = num(getf(f,10));
        case "GNS"
            cur.numSV_gns = num(getf(f,8));
            if isnan(cur.lat)
                cur.lat = dm_to_deg(getf(f,3),getf(f,4));
                cur.lon = dm_to_deg(getf(f,5),getf(f,6));
            end
        case "GSA"
            % All per-constellation GSA lines of an epoch repeat the same
            % solution DOPs; keep the first finite set.
            if isnan(cur.vdop)
                cur.pdop = num(getf(f,16));
                cur.hdop_gsa = num(getf(f,17));
                cur.vdop = num(getf(f,18));
            end
            % Satellite list, in the order the GRS residuals will follow.
            svs = arrayfun(@(k) num(getf(f,k)),4:15);
            cur.gsa_sys(end+1,1) = num(getf(f,19));
            cur.gsa_svs{end+1,1} = svs(:);
        case "GSV"
            [svid,elev,snr] = parse_gsv(f);
            k = numel(svid);
            if k > 0
                cur.sv_talker(end+1:end+k,1) = repmat(talker,k,1);
                cur.sv_id(end+1:end+k,1) = svid;
                cur.sv_elev(end+1:end+k,1) = elev;
                cur.sv_snr(end+1:end+k,1) = snr;
            end
        case "GST"
            cur.gst_rangeRms = num(getf(f,3));
            cur.gst_stdLat   = num(getf(f,7));
            cur.gst_stdLon   = num(getf(f,8));
            cur.gst_stdAlt   = num(getf(f,9));
        case "GRS"
            % Residual j belongs to satellite j of the GSA sentence with the
            % same systemId, so positions must be preserved, not compacted.
            r = arrayfun(@(k) num(getf(f,k)),4:15);
            cur.grs_sys(end+1,1) = num(getf(f,16));
            cur.grs_res{end+1,1} = r(:);
        case "GBS"
            cur.gbs_errLat = num(getf(f,3));
            cur.gbs_errLon = num(getf(f,4));
            cur.gbs_errAlt = num(getf(f,5));
            if strlength(getf(f,6)) > 0
                cur.gbs_faultSv = cur.gbs_faultSv + 1;   % a satellite is flagged
            end
        case "RMC"
            cur.speed_knots = num(getf(f,8));
            cur.dateStr = getf(f,10);
        case "VTG"
            cur.speed_kmh = num(getf(f,8));
    end
end
if ~isempty(cur), nE = nE+1; epochs{nE} = cur; end
epochs = epochs(1:nE);

E = epochs_to_table(epochs,opts.residualSignal);
end

% =========================================================================
function s = new_epoch(t)
s = struct('tod',t,'lat',NaN,'lon',NaN,'alt',NaN,'fixQual',NaN, ...
    'numSV_gga',NaN,'numSV_gns',NaN,'hdop_gga',NaN,'hdop_gsa',NaN, ...
    'pdop',NaN,'vdop',NaN, ...
    'gst_rangeRms',NaN,'gst_stdLat',NaN,'gst_stdLon',NaN,'gst_stdAlt',NaN, ...
    'gbs_errLat',NaN,'gbs_errLon',NaN,'gbs_errAlt',NaN,'gbs_faultSv',0, ...
    'speed_knots',NaN,'speed_kmh',NaN,'dateStr',"", ...
    'gsa_sys',zeros(0,1),'gsa_svs',{cell(0,1)}, ...
    'grs_sys',zeros(0,1),'grs_res',{cell(0,1)}, ...
    'sv_talker',strings(0,1),'sv_id',zeros(0,1),'sv_elev',nan(0,1),'sv_snr',nan(0,1));
end

function v = getf(f,i)
if i <= numel(f), v = f(i); else, v = ""; end
end

function x = num(s)
if strlength(s)==0, x = NaN; else, x = str2double(s); end
end

function t = tod_seconds(s)
% "hhmmss.ss" -> seconds since midnight UTC
if strlength(s) < 6, t = NaN; return, end
v = str2double(s);
if isnan(v), t = NaN; return, end
hh = floor(v/10000);
mm = floor((v-hh*10000)/100);
ss = v - hh*10000 - mm*100;
t = hh*3600 + mm*60 + ss;
end

function d = dm_to_deg(val,hemi)
% NMEA ddmm.mmmm -> signed decimal degrees
if strlength(val)==0, d = NaN; return, end
v = str2double(val);
if isnan(v), d = NaN; return, end
deg = floor(v/100);
minutes = v - deg*100;
d = deg + minutes/60;
if any(hemi == ["S","W"]), d = -d; end
end

function [svid,elev,snr] = parse_gsv(f)
% $xxGSV,numMsg,msgNum,numSV,(sv,elev,az,snr) x N [,signalId]
nf = numel(f);
body = nf - 4;                 % fields after the 3 header fields
if mod(body,4) == 1, body = body - 1; end   % trailing signalId
k = floor(body/4);
svid = zeros(k,1); elev = nan(k,1); snr = nan(k,1);
for j = 1:k
    b = 4 + (j-1)*4;
    svid(j) = num(getf(f,b+1));
    elev(j) = num(getf(f,b+2));
    snr(j)  = num(getf(f,b+4));
end
keep = isfinite(svid);
svid = svid(keep); elev = elev(keep); snr = snr(keep);
end

function r = per_satellite_residuals(e,mode)
%PER_SATELLITE_RESIDUALS Collapse GRS residuals to one value per satellite.
%
% A satellite is reported once per tracked signal (L1, L2, ...), so pooling raw
% GRS fields would count the same satellite several times and inflate both
% PR_RMS and any fault count. NMEA aligns residual j of a GRS sentence with
% satellite j of the GSA sentence carrying the same systemId, so the residuals
% are mapped back to satellite IDs and reduced to the worst residual per
% satellite.
r = zeros(0,1);
if isempty(e.grs_res), return, end

sysIds = unique(e.grs_sys(isfinite(e.grs_sys)));
keys = zeros(0,1); vals = zeros(0,1);

for s = sysIds(:)'
    % satellite list for this constellation (GSA may span several sentences)
    svs = zeros(0,1);
    for k = find(e.gsa_sys == s)'
        svs = [svs; e.gsa_svs{k}]; %#ok<AGROW>
    end
    svs = svs(isfinite(svs));
    if isempty(svs), continue, end

    sentences = find(e.grs_sys == s)';
    if mode == "primary" && ~isempty(sentences)
        sentences = sentences(1);   % primary signal only
    end
    for k = sentences
        res = e.grs_res{k};
        m = min(numel(res),numel(svs));
        for j = 1:m
            if ~isfinite(res(j)), continue, end
            key = s*1000 + svs(j);
            hit = find(keys == key,1);
            if isempty(hit)
                keys(end+1,1) = key;      %#ok<AGROW>
                vals(end+1,1) = res(j);   %#ok<AGROW>
            elseif abs(res(j)) > abs(vals(hit))
                vals(hit) = res(j);       % keep the worst residual for the SV
            end
        end
    end
end
r = vals;
end

function T = epochs_to_table(epochs,residualSignal)
n = numel(epochs);
tod = nan(n,1); lat = nan(n,1); lon = nan(n,1); alt = nan(n,1);
fixQual = nan(n,1); numSV_gga = nan(n,1); numSV_gns = nan(n,1);
hdop = nan(n,1); vdop = nan(n,1); pdop = nan(n,1);
stdLat = nan(n,1); stdLon = nan(n,1); stdAlt = nan(n,1); rangeRms = nan(n,1);
speed = nan(n,1); gbsFault = zeros(n,1);
prRms = nan(n,1); nRes = zeros(n,1);
cnoMean = nan(n,1); cnoStd = nan(n,1); cnoGap = nan(n,1);
nTracked = zeros(n,1); nVisible = zeros(n,1); lowElev = nan(n,1);
resid = cell(n,1); dateStr = strings(n,1);
svElev = cell(n,1); svSnr = cell(n,1);   % per-satellite raw values, for redefinition studies

for i = 1:n
    e = epochs{i};
    tod(i)=e.tod; lat(i)=e.lat; lon(i)=e.lon; alt(i)=e.alt;
    fixQual(i)=e.fixQual; numSV_gga(i)=e.numSV_gga; numSV_gns(i)=e.numSV_gns;
    if isfinite(e.hdop_gga), hdop(i)=e.hdop_gga; else, hdop(i)=e.hdop_gsa; end
    vdop(i)=e.vdop; pdop(i)=e.pdop;
    stdLat(i)=e.gst_stdLat; stdLon(i)=e.gst_stdLon; stdAlt(i)=e.gst_stdAlt;
    rangeRms(i)=e.gst_rangeRms; gbsFault(i)=e.gbs_faultSv;
    dateStr(i)=e.dateStr;

    if isfinite(e.speed_kmh),        speed(i)=e.speed_kmh/3.6;
    elseif isfinite(e.speed_knots),  speed(i)=e.speed_knots*0.514444;
    end

    r = per_satellite_residuals(e,residualSignal);
    resid{i} = r;
    nRes(i) = numel(r);
    if ~isempty(r), prRms(i) = sqrt(mean(r.^2)); end

    % Deduplicate satellites: the same SV is repeated once per signal (L1/L2/..)
    % and again with a blank C/N0 in the "signal 0" visibility group. Keep one
    % row per (constellation,SV), preferring the strongest reported C/N0.
    if ~isempty(e.sv_id)
        key = e.sv_talker + "_" + string(e.sv_id);
        [ukey,~,g] = unique(key,'stable');
        el = nan(numel(ukey),1); sn = nan(numel(ukey),1);
        for j = 1:numel(ukey)
            m = (g==j);
            el(j) = max(e.sv_elev(m),[],'omitnan');
            s = e.sv_snr(m);
            if any(isfinite(s)), sn(j) = max(s,[],'omitnan'); end
        end
        nVisible(i) = numel(ukey);
        tr = isfinite(sn) & sn > 0;
        nTracked(i) = sum(tr);
        svElev{i} = el(tr); svSnr{i} = sn(tr);
        if any(tr)
            cnoMean(i) = mean(sn(tr));
            cnoStd(i)  = std(sn(tr),0);
            cnoGap(i)  = max(sn(tr)) - min(sn(tr));
            eltr = el(tr);
            if any(isfinite(eltr))
                lowElev(i) = mean(eltr(isfinite(eltr)) < 15);
            end
        end
    end
end

T = table(tod,lat,lon,alt,fixQual,numSV_gga,numSV_gns,hdop,vdop,pdop, ...
    stdLat,stdLon,stdAlt,rangeRms,speed,prRms,nRes,gbsFault, ...
    cnoMean,cnoStd,cnoGap,nTracked,nVisible,lowElev,dateStr, ...
    'VariableNames',{'tod','lat','lon','alt','fixQuality','numSV_gga','numSV_gns', ...
    'hDOP','vDOP','pDOP','stdLat','stdLon','stdAlt','gstRangeRms','speed_mps', ...
    'PR_RMS','numResiduals','gbsFaultCount','CNO_mean','CNO_std','CNO_gap', ...
    'numTracked','numVisible','low_elev_ratio','dateStr'});
T.residuals = resid;
T.svElev = svElev;
T.svSnr = svSnr;
end
