function scan = readHokuyo30m(binFile,cfg)
%READHOKUYO30M Read NCLT Hokuyo UTM-30LX binary stream.
% Official devkit format: uint64 timestamp + 1081 uint16 ranges.
% Range conversion: raw*0.005 - 100 m; raw zero is invalid.

numHits = 1081;
recordBytes = 8 + 2*numHits;

info = dir(binFile);
n = floor(info.bytes/recordBytes);
if n < 1, error("No Hokuyo records found in %s",binFile); end

fid = fopen(binFile,"rb","ieee-le");
if fid < 0, error("Cannot open %s",binFile); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

utime = zeros(n,1,"uint64");
ranges = nan(numHits,n,"single");

for k = 1:n
    t = fread(fid,1,"uint64=>uint64");
    if isempty(t)
        utime = utime(1:k-1);
        ranges = ranges(:,1:k-1);
        break;
    end
    raw = fread(fid,numHits,"uint16=>uint16");
    if numel(raw) < numHits
        utime = utime(1:k-1);
        ranges = ranges(:,1:k-1);
        break;
    end
    utime(k) = t;
    r = single(raw)*single(0.005) - single(100.0);
    r(raw==0) = NaN;
    r(r<cfg.lidar.minRange | r>cfg.lidar.maxRange) = NaN;
    ranges(:,k) = r;
end

scan.time = double(utime)*1e-6;
scan.ranges = ranges;
scan.angles = deg2rad((-135:0.25:135)).';
end
