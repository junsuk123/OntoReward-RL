function a=expertController(x,cfg)
%EXPERTCONTROLLER PX4-compatible behavior policy without simulator feedforward.
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10));
vzDes=-min(0.70,max(0.16,0.18+0.18*p(3)));
azCmd=2.2*(vzDes-v(3));
collective=azCmd/(cfg.sim.g*cfg.rl.collectiveSpan);
ax=-1.15*p(1)-0.95*v(1); ay=-1.15*p(2)-0.95*v(2);
pitchDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,ax/cfg.sim.g));
rollDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,-ay/cfg.sim.g));
a=[collective;rollDes/cfg.rl.maxRollPitch;pitchDes/cfg.rl.maxRollPitch; ...
    -0.6*rpy(3)/cfg.rl.maxYawRate];
a=max(-1,min(1,a));
end
