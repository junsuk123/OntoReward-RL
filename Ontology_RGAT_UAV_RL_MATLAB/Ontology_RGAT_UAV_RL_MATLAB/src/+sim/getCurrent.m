function cur = getCurrent(env,cfg)
meas=sensor.observe(env.x,env.lastDiag,cfg);
sem=semantic.computeFeatures(env.x,env.lastDiag,meas,env.prevSem,cfg);
graph=semantic.buildOntologyGraph(sem,cfg);
cur=struct('meas',meas,'sem',sem,'graph',graph,'obs',sim.makeObservation(meas,sem,cfg));
end
