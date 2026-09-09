function pts = rangesToPoints(ranges,angles,cfg)
%RANGESTOPOINTS Convert Hokuyo scan to body-like 2-D coordinates.
% NCLT UTM30-LX mount has roll 180 deg, so planar y is flipped.

idx = 1:cfg.lidar.beamStride:numel(ranges);
r = double(ranges(idx));
a = double(angles(idx));
valid = isfinite(r) & r>=cfg.lidar.minRange & r<=cfg.lidar.maxRange;

r = r(valid);
a = a(valid);

x = r.*cos(a) + 0.28;  % x_body,h30 from NCLT calibration table
y = -r.*sin(a);        % roll=180 deg flips planar y sign
pts = [x(:),y(:)];

if size(pts,1) > cfg.lidar.maxPointsICP
    sel = round(linspace(1,size(pts,1),cfg.lidar.maxPointsICP));
    pts = pts(sel,:);
end
end
