function G = buildOntologyGraph()
%BUILDONTOLOGYGRAPH Typed prior graph for the UGV localization ontology.
%
% Observed nodes 1:12 come from the schema feature table.
% Latent nodes 13:16 are predicted by graph message passing.

G.nodeNames = [ ...
    "GPSFix","NumSV","GPSInnovation","GPSSpeedResidual", ...
    "LidarValidRatio","LidarInlierRatio","LidarICPRMSE","LidarStructure", ...
    "OdomCovXY","OdomYawVar","Dynamics","OpenSkyScore", ...
    "GNSSReliability","LiDARReliability","OdomReliability","LocalizationDifficulty"];

G.relationNames = ["self","indicates","contextSupports","supports","degrades"];
SELF=1; IND=2; CTX=3; SUP=4; DEG=5;

src=[]; dst=[]; rel=[];

% Self loops.
for i=1:numel(G.nodeNames)
    src(end+1)=i; dst(end+1)=i; rel(end+1)=SELF; %#ok<AGROW>
end

% GNSS evidence -> GNSS reliability.
add([1 2],13,IND);
add([3 4],13,DEG);
add(12,13,CTX);

% LiDAR evidence -> LiDAR reliability.
add([5 6 8],14,IND);
add(7,14,DEG);
add([8 11],14,CTX);

% Odometry / dynamics -> odometry reliability.
add([9 10],15,DEG);
add(11,15,CTX);

% Sensor reliability -> localization difficulty.
add([13 14 15],16,SUP);
add(11,16,DEG);
add([3 7],16,DEG);

% Sort by destination so attention vectors are stable and easy to log.
[~,ord] = sortrows([dst(:),src(:)],[1 2]);
G.src = src(ord).';
G.dst = dst(ord).';
G.rel = rel(ord).';
G.numNodes = numel(G.nodeNames);
G.numRelations = numel(G.relationNames);
G.outputNodes = 13:16;

    function add(s,d,r)
        for z=s
            src(end+1)=z; dst(end+1)=d; rel(end+1)=r; %#ok<AGROW>
        end
    end
end
