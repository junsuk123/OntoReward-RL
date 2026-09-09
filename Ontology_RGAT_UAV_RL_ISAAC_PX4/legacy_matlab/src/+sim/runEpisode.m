function log=runEpisode(policy,rewardMode,rgatModel,seed,cfg,showRealtime)
if nargin<6, showRealtime=false; end
env=sim.resetState(seed,cfg); cleanup=onCleanup(@()safeDisarm(env));
T=cfg.sim.maxSteps;
log.t=zeros(1,T); log.x=zeros(17,T); log.a=zeros(4,T); log.r=zeros(1,T);
log.wind=zeros(3,T); log.aeroF=zeros(1,T); log.tilt=zeros(1,T);
log.aeroForce=zeros(3,T);
log.power=zeros(1,T); log.phi=nan(1,T);
log.graphX=zeros(cfg.ontology.inDim,cfg.ontology.nNodes,T);
log.graphTemplate=[];
if showRealtime, mon=viz.RealtimeMonitor(cfg); else, mon=[]; end
status='timeout';
for k=1:T
    cur=sim.getCurrent(env,cfg);
    if k==1, log.graphTemplate=cur.graph; end
    a=training.policyAction(policy,cur,env,cfg);
    [env2,~,r,done,info]=sim.step(env,a,cur,rewardMode,rgatModel,cfg);
    log.t(k)=env.t; log.x(:,k)=env.x; log.a(:,k)=a; log.r(k)=r;
    log.wind(:,k)=env.lastDiag.meanWindI;
    log.aeroF(k)=env.lastDiag.aeroForceMag;
    log.aeroForce(:,k)=env.lastDiag.aeroForceI;
    log.tilt(k)=cur.sem.tilt; log.graphX(:,:,k)=cur.graph.X;
    if strcmpi(rewardMode,'proposed'), log.phi(k)=info.rewardParts.phi0; end
    if showRealtime && (mod(k,cfg.sim.monitorEvery)==0 || done)
        update(mon,log,k,env2,sim.getCurrent(env2,cfg),info); drawnow limitrate;
    end
    env=env2; status=info.status;
    if done, break; end
end
fields={'t','r','aeroF','tilt','power','phi'};
for i=1:numel(fields), log.(fields{i})=log.(fields{i})(1:k); end
log.x=log.x(:,1:k); log.a=log.a(:,1:k); log.wind=log.wind(:,1:k);
log.aeroForce=log.aeroForce(:,1:k);
log.graphX=log.graphX(:,:,1:k);
final=env.x; rpy=mathx.quatToEulerZYX(final(7:10));
log.metrics=struct('seed',seed,'status',status, ...
    'success',double(strcmp(status,'success')), ...
    'unsafe',double(any(strcmp(status,{'unsafe_touchdown','flight_failure'}))), ...
    'timeout',double(strcmp(status,'timeout')),'steps',k,'return',sum(log.r), ...
    'touchdownXY',norm(final(1:2)),'touchdownVz',abs(final(6)), ...
    'maxTilt',max(log.tilt),'maxAeroForce',max(log.aeroF), ...
    'energyJ',NaN,'finalTilt',norm(rpy(1:2)),'backend','Isaac/PX4');
clear cleanup;
end

function safeDisarm(env)
% Release the bridge as well as disarming: the next episode needs the local
% UDP port back, and MATLAB will not collect the handle in time on its own.
if ~isfield(env,'bridge') || ~isvalid(env.bridge)
    return;
end
try
    env.bridge.disarm();
catch
end
delete(env.bridge);
end
