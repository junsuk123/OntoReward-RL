function omegaCmd = actionToRotorCmd(action,x,cfg)
%ACTIONTOROTORCMD Convert normalized RL action to attitude/thrust command.
a=max(-1,min(1,action(:)));
rpy=mathx.quatToEulerZYX(x(7:10)); omega=x(11:13);
Tdes=cfg.drone.mass*cfg.sim.g*(1+cfg.rl.collectiveSpan*a(1));
Tdes=max(0.15*cfg.drone.mass*cfg.sim.g,min(cfg.drone.maxTotalThrust,Tdes));
rpyDes=[cfg.rl.maxRollPitch*a(2); cfg.rl.maxRollPitch*a(3); rpy(3)];
yawRateDes=cfg.rl.maxYawRate*a(4);
err=[mathx.wrapPi(rpyDes(1)-rpy(1)); mathx.wrapPi(rpyDes(2)-rpy(2)); 0];
rateDes=[0;0;yawRateDes];
M=cfg.rl.KpAtt.*err + cfg.rl.KdAtt.*(rateDes-omega);
% Conservative torque saturation for numerical robustness.
M=max([-2.0;-2.0;-1.0],min([2.0;2.0;1.0],M));
omegaCmd=prop.mixWrenchToOmega(Tdes,M,cfg);
end
