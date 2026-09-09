function data = generateSyntheticDemo(cfg)
%GENERATESYNTHETICDEMO Realistic software smoke-test with environment transitions.
% This is NOT a substitute for NCLT results.

T = min(cfg.data.maxSamples,1200);
if ~isfinite(T),T=1200;end
dt=cfg.data.dt;
time=(0:T-1)'*dt;

% Smooth UGV trajectory.
v=2.0+0.35*sin(0.015*(1:T)');
yawRate=0.025*sin(0.012*(1:T)') + 0.06*(mod((1:T)',260)>190);
yaw=cumsum(yawRate*dt);
x=cumsum(v.*cos(yaw)*dt);
y=cumsum(v.*sin(yaw)*dt);
gt=[x y wrapAngle(yaw)];

% Context: 1 open, 2 urban canyon, 3 tunnel, 4 lidar-poor.
ctx=ones(T,1);
ctx(round(.25*T):round(.48*T))=2;
ctx(round(.48*T):round(.66*T))=3;
ctx(round(.76*T):round(.91*T))=4;

% Odometry drift.
odom=gt;
odom(:,1)=odom(:,1)+cumsum(0.02*randn(T,1));
odom(:,2)=odom(:,2)+cumsum(0.02*randn(T,1));
odom(:,3)=wrapAngle(odom(:,3)+cumsum(deg2rad(0.08)*randn(T,1)));
odomCovXY=0.08+0.05*(ctx==2)+0.12*(ctx==3);
odomYawVar=deg2rad(1.5+1.0*(ctx==3)).^2;

% GPS.
gpsSigma=0.8*ones(T,1);
gpsSigma(ctx==2)=6;
gpsSigma(ctx==3)=20;
gpsSigma(ctx==4)=1.5;
gpsValid=ctx~=3;
gpsXY=gt(:,1:2)+gpsSigma.*randn(T,2);
gpsXY(~gpsValid,:)=NaN;
gpsFix=3*ones(T,1); gpsFix(~gpsValid)=1;
numSV=11*ones(T,1); numSV(ctx==2)=6; numSV(ctx==3)=0; numSV(ctx==4)=9;
gpsSpeed=v+0.15*randn(T,1);

% LiDAR odometry: good except feature-poor segment.
lidarPose=gt;
ls=0.18*ones(T,1);
ls(ctx==4)=2.2;
ls(ctx==3)=0.30;
lidarPose(:,1)=lidarPose(:,1)+cumsum(ls.*0.03.*randn(T,1));
lidarPose(:,2)=lidarPose(:,2)+cumsum(ls.*0.03.*randn(T,1));
lidarPose(:,3)=wrapAngle(lidarPose(:,3)+cumsum(ls.*deg2rad(0.08).*randn(T,1)));

lidarValidRatio=0.88*ones(T,1); lidarValidRatio(ctx==4)=0.35;
lidarIcpInlier=0.78*ones(T,1); lidarIcpInlier(ctx==4)=0.20;
lidarIcpRmse=0.18*ones(T,1); lidarIcpRmse(ctx==4)=1.3;
lidarStructure=0.82*ones(T,1); lidarStructure(ctx==4)=0.12;

data.time=time;
data.utimeSec=time;
data.gt=gt;
data.odom=odom;
data.gpsXY=gpsXY;
data.gpsValid=gpsValid;
data.gpsFix=gpsFix;
data.numSV=numSV;
data.gpsSpeed=gpsSpeed;
data.yawRate=yawRate;
data.accelNorm=9.81+0.3*randn(T,1);
data.odomCovXY=odomCovXY;
data.odomYawVar=odomYawVar;
data.lidarPose=lidarPose;
data.lidarValidRatio=lidarValidRatio;
data.lidarStructure=lidarStructure;
data.lidarIcpInlier=lidarIcpInlier;
data.lidarIcpRmse=lidarIcpRmse;
data.source="Synthetic smoke test";
data.context=ctx;

data=buildSchemaFeatures(data,cfg);
end
