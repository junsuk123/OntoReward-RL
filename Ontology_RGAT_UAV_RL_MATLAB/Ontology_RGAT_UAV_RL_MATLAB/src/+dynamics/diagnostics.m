function d = diagnostics(x,omegaCmd,t,cfg)
[~,e]=dynamics.eom(x,omegaCmd,t,cfg);
d=e;
d.time=t;
d.meanWindI=e.aero.meanWindI;
d.stdWind=e.aero.stdWind;
d.aeroForceMag=norm(e.aero.Fb);
d.aeroMomentMag=norm(e.aero.Mb);
d.totalThrust=sum(e.rotor.thrust);
d.inducedPower=e.rotor.inducedPower;
end
