function xyz = gpsToLocalNCLT(latRad,lonRad,alt)
%GPSTOLOCALNCLT NCLT paper's local GPS linearization.
% x = sin(lat-lat0)*r_ns
% y = sin(lon-lon0)*r_ew*cos(lat0)
% z = alt0-alt

lat0 = deg2rad(42.293227);
lon0 = deg2rad(-83.709657);
alt0 = 270.0;
re = 6378135.0;
rp = 6356750.0;

den = ((re*cos(lat0))^2 + (rp*sin(lat0))^2);
rns = (re*rp)^2 / (den^(3/2));
rew = re^2 / sqrt(den);

x = sin(latRad-lat0) .* rns;
y = sin(lonRad-lon0) .* rew .* cos(lat0);
z = alt0 - alt;
xyz = [x(:),y(:),z(:)];
end
