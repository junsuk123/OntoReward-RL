function gain = groundEffectGain(z,cfg)
%GROUNDEFFECTGAIN Rotor thrust augmentation factor inside ground effect.
% Shared by the plant (prop.rotorForces) and by any controller that feeds the
% effect forward (control.expertController), so the two can never drift apart.
if ~cfg.prop.groundEffect.enable
    gain=1; return;
end
h=max(z,cfg.sim.groundZ);
gain=1+(cfg.prop.groundEffect.maxGain-1)*exp(-(h/cfg.prop.groundEffect.heightScale)^2);
gain=min(cfg.prop.groundEffect.maxGain,max(1,gain));
end
