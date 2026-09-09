function m = computeMetrics(data,testIdx,result,cfg)
%COMPUTEMETRICS Accuracy, uncertainty, computational, and sampling metrics.

gt=data.gt(testIdx,:);
eXY=result.est(:,1:2)-gt(:,1:2);
posErr=sqrt(sum(eXY.^2,2));
yawErr=wrapAngle(result.est(:,3)-gt(:,3));

nees=nan(numel(testIdx),1);
for k=1:numel(testIdx)
    P=result.P(1:2,1:2,k);
    e=eXY(k,:).';
    if all(isfinite(P),"all") && rcond(P)>1e-10
        nees(k)=e.'*(P\e);
    end
end

m.PositionRMSE_m = sqrt(mean(posErr.^2,"omitnan"));
m.PositionMAE_m = mean(posErr,"omitnan");
m.PositionP95_m = percentileSimple(posErr,95);
m.YawRMSE_deg = rad2deg(sqrt(mean(yawErr.^2,"omitnan")));
m.FailureRate_pct = 100*mean(posErr>cfg.pf.failureThreshold);
m.MeanNEES2D = mean(nees,"omitnan");
m.MeanParticles = mean(result.N,"omitnan");
m.MaxParticles = max(result.N);
m.MeanRuntime_ms = mean(result.runtimeMs,"omitnan");
m.P95Runtime_ms = percentileSimple(result.runtimeMs,95);
m.RealTimeFactor = (1000*median(diff(data.time(testIdx))))/max(eps,m.MeanRuntime_ms);
end
