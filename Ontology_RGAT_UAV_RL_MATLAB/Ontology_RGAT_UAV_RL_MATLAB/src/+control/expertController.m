function a = expertController(x,cfg)
%EXPERTCONTROLLER Simple cascaded landing controller used only for data generation.
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10));
% Smooth vertical profile: target descent slows near ground.
vzDes=-min(0.70,max(0.16,0.18+0.18*p(3)));
azCmd=2.2*(vzDes-v(3));
% Feed-forward cancellation of the ground-effect thrust surplus. The vertical
% loop is proportional only, so without this term the surplus near the pad
% exactly balances the commanded thrust deficit and the drone settles into a
% hover it can never leave (stable equilibrium at z ~ heightScale).
gain=prop.groundEffectGain(p(3),cfg);
mg=cfg.drone.mass*cfg.sim.g;
Tdes=cfg.drone.mass*(cfg.sim.g+azCmd)/gain;
collective=(Tdes/mg-1)/cfg.rl.collectiveSpan;
% Horizontal PD acceleration -> small-angle roll/pitch commands.
ax=-1.15*p(1)-0.95*v(1);
ay=-1.15*p(2)-0.95*v(2);
pitchDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,ax/cfg.sim.g));
rollDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,-ay/cfg.sim.g));
a=[collective; rollDes/cfg.rl.maxRollPitch; pitchDes/cfg.rl.maxRollPitch; -0.6*rpy(3)/cfg.rl.maxYawRate];
a=max(-1,min(1,a));
end
