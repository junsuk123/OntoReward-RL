function y = observe(x,diagOut,cfg)
%OBSERVE Noisy navigation and abstract camera-marker measurements.
rpy=mathx.quatToEulerZYX(x(7:10));
y.pos=x(1:3)+cfg.sensor.posStd*randn(3,1);
y.vel=x(4:6)+cfg.sensor.velStd*randn(3,1);
y.rpy=rpy+cfg.sensor.angleStd*randn(3,1);
y.omega=x(11:13)+cfg.sensor.rateStd*randn(3,1);
xy=norm(y.pos(1:2)); tilt=norm(y.rpy(1:2)); range=max(y.pos(3),0);
windProxy=min(1,diagOut.aeroForceMag/max(1,cfg.drone.mass*cfg.sim.g));
p=cfg.sensor.marker.baseDetect ...
    *exp(-(range/cfg.sensor.marker.maxRange)^2) ...
    *exp(-(tilt/cfg.sensor.marker.tiltScale)^2) ...
    *exp(-(xy/cfg.sensor.marker.xyScale)^2) ...
    *exp(-(windProxy/cfg.sensor.marker.windScale)^2);
p=max(0.02,min(0.999,p));
y.markerProb=p;
y.markerDetected=rand<p;
% Continuous visual quality is preserved even when a Bernoulli dropout occurs.
y.markerQuality=p*(0.3+0.7*double(y.markerDetected));
end
