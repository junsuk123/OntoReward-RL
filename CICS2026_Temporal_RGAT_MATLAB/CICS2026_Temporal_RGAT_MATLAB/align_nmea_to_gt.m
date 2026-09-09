function [Em,Gm,posErr,enu] = align_nmea_to_gt(E,G,maxTimeGapS)
%ALIGN_NMEA_TO_GT Match NMEA epochs to ground-truth epochs on UTC time-of-day.
%
%   [Em,Gm,posErr,enu] = align_nmea_to_gt(E,G,maxTimeGapS)
%
% E is a parse_nmea_epochs table, G a read_urbannav_gt table. Each GT epoch is
% paired with the nearest NMEA epoch; pairs further apart than maxTimeGapS are
% dropped. Returns the matched rows plus the 2-D position error between the
% receiver fix and ground truth in a local ENU frame anchored at the first GT
% point, and that frame's coordinates.
%
% Shared by build_urbannav_features and any feature-definition study, so
% both always use identical epoch pairing.

arguments
    E table
    G table
    maxTimeGapS (1,1) double = 0.6
end

E = E(isfinite(E.tod) & isfinite(E.lat) & isfinite(E.lon),:);
if isempty(E)
    Em = E; Gm = G([],:); posErr = []; enu = struct(); return
end

% Unwrap a UTC midnight rollover so nearest-neighbour matching stays monotonic.
etod = unwrap_tod(E.tod);
gtod = unwrap_tod(G.tod);

% interp1 needs strictly increasing, unique sample points.
[etod,keep] = unique(etod,'stable');
E = E(keep,:);
[etod,ord] = sort(etod);
E = E(ord,:);

idx = interp1(etod,(1:numel(etod))',gtod,'nearest','extrap');
idx = idx(:);
ok = abs(gtod - etod(idx)) <= maxTimeGapS;

Gm = G(ok,:);
Em = E(idx(ok),:);
if isempty(Em), posErr = []; enu = struct(); return, end

[gE,gN] = geodetic_to_enu(Gm.lat,Gm.lon,Gm.lat(1),Gm.lon(1));
[rE,rN] = geodetic_to_enu(Em.lat,Em.lon,Gm.lat(1),Gm.lon(1));
posErr = hypot(rE-gE,rN-gN);
enu = struct('gnssEast',rE,'gnssNorth',rN,'gtEast',gE,'gtNorth',gN);
end

function u = unwrap_tod(t)
u = t(:);
d = diff(u);
j = find(d < -43200);          % midnight rollover
for k = 1:numel(j)
    u(j(k)+1:end) = u(j(k)+1:end) + 86400;
end
end

function [e,n] = geodetic_to_enu(lat,lon,lat0,lon0)
% WGS-84 geodetic -> local ENU (east/north only), no toolbox dependency.
a = 6378137.0; f = 1/298.257223563; e2 = f*(2-f);
d2r = pi/180;
lat = lat*d2r; lon = lon*d2r; lat0 = lat0*d2r; lon0 = lon0*d2r;
Rn = a./sqrt(1-e2*sin(lat0).^2);           % prime vertical radius
Rm = a*(1-e2)./(1-e2*sin(lat0).^2).^1.5;   % meridional radius
e = (lon-lon0).*cos(lat0).*Rn;
n = (lat-lat0).*Rm;
end
