function out = interpPose(tSrc,poseSrc,tq)
%INTERPPOSE Linear xy + unwrap/interpolate yaw.
%
% Non-finite samples are dropped before interpolation. NCLT logs occasionally
% contain corrupted rows -- groundtruth_2013-01-10.csv has a line of NUL bytes
% that parses to NaN -- and interp1 rejects non-finite sample points outright.

tSrc = tSrc(:);
good = isfinite(tSrc) & all(isfinite(poseSrc),2);
tSrc = tSrc(good);
poseSrc = poseSrc(good,:);

if numel(tSrc) < 2
    error("interpPose:InsufficientSamples", ...
        "Fewer than two finite pose samples remain after dropping corrupt rows.");
end

% interp1 requires strictly increasing, unique sample points.
[tSrc,ia] = unique(tSrc,"sorted");
poseSrc = poseSrc(ia,:);

out = zeros(numel(tq),3);
out(:,1) = interp1(tSrc,poseSrc(:,1),tq,"linear","extrap");
out(:,2) = interp1(tSrc,poseSrc(:,2),tq,"linear","extrap");
yaw = unwrap(poseSrc(:,3));
out(:,3) = wrapAngle(interp1(tSrc,yaw,tq,"linear","extrap"));
end
