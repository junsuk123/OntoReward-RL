function G = read_urbannav_gt(gtFile)
%READ_URBANNAV_GT Read a UrbanNav raw ground-truth trajectory file.
%
% The published raw GT files are fixed-column SPAN/Inertial-Explorer exports:
%
%   UTCTime  Week  GPSTime  Latitude(+/-D M S)  Longitude(+/-D M S)  H-Ell ...
%
% Latitude/Longitude occupy three whitespace-separated fields each (degrees,
% minutes, seconds), so the header has fewer tokens than the data rows and
% readtable cannot infer the layout. The rows are parsed positionally instead.
%
% Returns a table with utcTime (POSIX seconds), tod (seconds since UTC
% midnight), lat/lon in decimal degrees, height, and the quality flag Q.

arguments
    gtFile (1,1) string
end

lines = readlines(gtFile);
lines = strip(lines);
lines = lines(strlength(lines) > 0);

utc=[]; week=[]; gpst=[]; lat=[]; lon=[]; hEll=[]; q=[];
for i = 1:numel(lines)
    tok = split(lines(i));
    tok = tok(strlength(tok) > 0);
    if numel(tok) < 20, continue, end
    v = str2double(tok);
    if any(isnan(v(1:11))), continue, end   % header / unit rows

    utc(end+1,1)  = v(1);              %#ok<AGROW>
    week(end+1,1) = v(2);              %#ok<AGROW>
    gpst(end+1,1) = v(3);              %#ok<AGROW>
    lat(end+1,1)  = dms(v(4),v(5),v(6));   %#ok<AGROW>
    lon(end+1,1)  = dms(v(7),v(8),v(9));   %#ok<AGROW>
    hEll(end+1,1) = v(10);             %#ok<AGROW>
    q(end+1,1)    = v(end);            %#ok<AGROW>
end

if isempty(utc)
    error('read_urbannav_gt:noData','No parsable GT rows found in %s',gtFile);
end

tod = mod(utc,86400);
G = table(utc,tod,week,gpst,lat,lon,hEll,q, ...
    'VariableNames',{'utcTime','tod','gpsWeek','gpsTime','lat','lon','height','Q'});
end

function d = dms(deg,mn,sec)
% Sign lives on the degrees field; minutes/seconds are magnitudes.
s = sign(deg); if s == 0, s = 1; end
d = s*(abs(deg) + mn/60 + sec/3600);
end
