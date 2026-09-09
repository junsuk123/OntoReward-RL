function F = derive_urbannav_features(Em,opts)
%DERIVE_URBANNAV_FEATURES Compute the twelve model inputs from aligned NMEA epochs.
%
%   F = derive_urbannav_features(Em,opts)
%
% Em is the matched NMEA table from align_nmea_to_gt. Returns a table with
% exactly the twelve feature columns in cfg.featureNames order.
%
% Four of the twelve are not fully specified by the prior presentation, so their
% definitions are options rather than hard-coded choices. They are collected
% here, in one place, because a definition that looks harmless can decide
% whether the features transfer between scenarios:
%
%   cnoGapMode        how the C/N0 spread is summarised. "maxmin" is the literal
%                     reading but is set by a single weakest satellite, so it is
%                     dominated by outliers; the percentile and top/bottom-k
%                     variants are robust alternatives.
%   lowElevDeg        elevation threshold for "low elevation".
%   prRmsMode         how per-satellite residuals are summarised. Residuals are
%                     heavy-tailed (single-digit to hundreds of metres), so a
%                     plain RMS is an outlier statistic.
%   faultCountMode    "count" is the literal reading, but a raw count scales with
%                     how many satellites are tracked, which itself differs
%                     between scenarios; "ratio" divides it out.

arguments
    Em table
    opts.numSVSource (1,1) string {mustBeMember(opts.numSVSource,["gns","gga"])} = "gns"
    opts.lowElevDeg (1,1) double = 15
    opts.faultResidualM (1,1) double = 10
    opts.cnoGapMode (1,1) string ...
        {mustBeMember(opts.cnoGapMode,["maxmin","p90p10","max_minus_median","top3_minus_bot3"])} = "maxmin"
    opts.prRmsMode (1,1) string ...
        {mustBeMember(opts.prRmsMode,["rms","median_abs","log_rms","trimmed_rms"])} = "rms"
    opts.faultCountMode (1,1) string ...
        {mustBeMember(opts.faultCountMode,["count","ratio"])} = "count"
end

n = height(Em);

% ---- directly reported quantities ---------------------------------------
if opts.numSVSource == "gns"
    numSV = Em.numSV_gns;
    numSV(~isfinite(numSV)) = Em.numSV_gga(~isfinite(numSV));
else
    numSV = Em.numSV_gga;
end

hAcc = hypot(Em.stdLat,Em.stdLon);
vAcc = Em.stdAlt;

% ---- C/N0 and elevation summaries ---------------------------------------
cnoMean = nan(n,1); cnoStd = nan(n,1); cnoGap = nan(n,1); lowElev = nan(n,1);
for i = 1:n
    s = Em.svSnr{i}; e = Em.svElev{i};
    s = s(isfinite(s));
    if ~isempty(s)
        cnoMean(i) = mean(s);
        cnoStd(i)  = std(s,0);
        cnoGap(i)  = cno_gap(s,opts.cnoGapMode);
    end
    e = e(isfinite(e));
    if ~isempty(e)
        lowElev(i) = mean(e < opts.lowElevDeg);
    end
end

% ---- residual-derived features ------------------------------------------
prRms = nan(n,1); faultCount = zeros(n,1);
for i = 1:n
    r = Em.residuals{i};
    r = r(isfinite(r));
    if ~isempty(r)
        prRms(i) = pr_summary(r,opts.prRmsMode);
        c = sum(abs(r) > opts.faultResidualM);
        if opts.faultCountMode == "ratio"
            faultCount(i) = c/numel(r);
        else
            faultCount(i) = c;
        end
    end
end
if opts.faultCountMode == "count"
    faultCount = faultCount + Em.gbsFaultCount;
else
    denom = max(cellfun(@numel,Em.residuals),1);
    faultCount = faultCount + Em.gbsFaultCount./denom(:);
end

F = table(numSV,Em.hDOP,Em.vDOP,hAcc,vAcc,Em.speed_mps, ...
    cnoMean,cnoStd,cnoGap,lowElev,prRms,faultCount, ...
    'VariableNames',{'numSV','hDOP','vDOP','hAcc','vAcc','gSpeed', ...
    'CNO_mean','CNO_std','CNO_gap','low_elev_ratio','PR_RMS','Fault_SVID_count'});
end

% =========================================================================
function g = cno_gap(s,mode)
switch mode
    case "maxmin"
        g = max(s) - min(s);
    case "p90p10"
        g = prctile(s,90) - prctile(s,10);
    case "max_minus_median"
        g = max(s) - median(s);
    case "top3_minus_bot3"
        v = sort(s,'descend');
        k = min(3,numel(v));
        g = mean(v(1:k)) - mean(v(end-k+1:end));
end
end

function p = pr_summary(r,mode)
switch mode
    case "rms"
        p = sqrt(mean(r.^2));
    case "median_abs"
        p = median(abs(r));
    case "log_rms"
        p = log1p(sqrt(mean(r.^2)));
    case "trimmed_rms"
        a = sort(abs(r));
        k = max(1,floor(0.9*numel(a)));    % drop the worst 10%
        p = sqrt(mean(a(1:k).^2));
end
end
