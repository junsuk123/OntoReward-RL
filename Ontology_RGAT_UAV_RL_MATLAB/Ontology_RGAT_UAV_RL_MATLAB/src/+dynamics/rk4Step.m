function x1 = rk4Step(x,omegaCmd,t,cfg)
dt=cfg.sim.dt;
k1=dynamics.eom(x,omegaCmd,t,cfg);
k2=dynamics.eom(x+0.5*dt*k1,omegaCmd,t+0.5*dt,cfg);
k3=dynamics.eom(x+0.5*dt*k2,omegaCmd,t+0.5*dt,cfg);
k4=dynamics.eom(x+dt*k3,omegaCmd,t+dt,cfg);
x1=x+dt*(k1+2*k2+2*k3+k4)/6;
x1(7:10)=mathx.quatNormalize(x1(7:10));
x1(14:17)=max(cfg.prop.omegaMin,min(cfg.prop.omegaMax,x1(14:17)));
end
