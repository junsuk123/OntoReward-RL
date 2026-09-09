function [loss,G] = ppoCriticGradients(C,obs,ret,cfg)
v=training.criticForward(C,obs);
loss=cfg.ppo.valueCoef*mean((v-ret).^2,'all');
[G.W1,G.b1,G.W2,G.b2,G.Wv,G.bv]=dlgradient(loss,C.W1,C.b1,C.W2,C.b2,C.Wv,C.bv);
end
