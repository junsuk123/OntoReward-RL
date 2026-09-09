function env = resetState(seed,cfg)
%RESETSTATE Randomized landing initial condition and steady hover motor speed.
rng(seed,'twister');
p=[1.2*randn;1.2*randn;3.4+1.4*rand];
v=[0.15*randn;0.15*randn;-0.15*rand];
rpy=[deg2rad(4)*randn;deg2rad(4)*randn;deg2rad(12)*randn];
q=mathx.eulerToQuat(rpy);
omega=deg2rad(3)*randn(3,1);
omegaHover=sqrt(cfg.drone.mass*cfg.sim.g/(4*cfg.prop.kT));
wr=omegaHover*ones(4,1);
x=[p;v;q;omega;wr];
hovCmd=omegaHover*ones(4,1);
d=dynamics.diagnostics(x,hovCmd,0,cfg);
env=struct('x',x,'t',0,'step',0,'prevSem',[],'lastDiag',d,'done',false,'seed',seed);
end
