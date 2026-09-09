function atm = airProperties(z,cfg)
%AIRPROPERTIES Low-altitude ISA density + Sutherland viscosity.
z = max(-100,min(11000,z));
T = cfg.aero.T0 - cfg.aero.lapse*z;
p = cfg.aero.p0*(T/cfg.aero.T0)^(cfg.sim.g/(cfg.aero.Rair*cfg.aero.lapse));
rho = p/(cfg.aero.Rair*T);
mu = cfg.aero.mu0*(T/cfg.aero.T0)^(3/2)*(cfg.aero.T0+cfg.aero.Suth)/(T+cfg.aero.Suth);
atm = struct('rho',rho,'mu',mu,'T',T,'p',p);
end
