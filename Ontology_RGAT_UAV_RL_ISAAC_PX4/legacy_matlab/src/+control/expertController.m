function a=expertController(x,cfg,cur)
%EXPERTCONTROLLER PX4-compatible behavior policy without simulator feedforward.
%   The state is pad-relative, so the same PD that used to hold a fixed pad now
%   chases a moving one: driving pad-relative position and velocity to zero *is*
%   tracking the deck. Two external additions:
%
%     - the deck's own velocity is fed forward, so the controller leads the
%       target instead of only reacting to the error it has already accrued;
%     - the descent is compressed when the energy margin is thin, so the
%       demonstrations the R-GAT dataset is labelled from actually contain
%       successful low-reserve landings. Without this every low-battery episode
%       is a negative example and the potential has nothing to learn from.
p=x(1:3); v=x(4:6); rpy=mathx.quatToEulerZYX(x(7:10));
[padVel,margin]=context(cur);
% Nominal profile, then urgency: a thin margin buys descent rate, bounded so the
% controller never asks for a touchdown speed the criteria would reject.
vzNominal=min(0.70,max(0.16,0.18+0.18*p(3)));
urgency=min(1,max(0,1-margin));
vzDes=-min(0.95*cfg.criteria.vz,vzNominal*(1+0.6*urgency));
azCmd=2.2*(vzDes-v(3));
collective=azCmd/(cfg.sim.g*cfg.rl.collectiveSpan);
% Horizontal PD on the pad-relative error, plus deck-velocity feed-forward.
ax=-1.15*p(1)-0.95*v(1)+0.55*padVel(1);
ay=-1.15*p(2)-0.95*v(2)+0.55*padVel(2);
pitchDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,ax/cfg.sim.g));
rollDes=max(-cfg.rl.maxRollPitch,min(cfg.rl.maxRollPitch,-ay/cfg.sim.g));
a=[collective;rollDes/cfg.rl.maxRollPitch;pitchDes/cfg.rl.maxRollPitch; ...
    -0.6*rpy(3)/cfg.rl.maxYawRate];
a=max(-1,min(1,a));
end

function [padVel,margin] = context(cur)
%CONTEXT Deck velocity and energy margin, defaulting to the static full-pack
%   case so this controller still runs against a link that reports neither.
padVel=[0;0;0]; margin=1;
if nargin<1 || isempty(cur) || ~isstruct(cur), return; end
if isfield(cur,'meas') && isfield(cur.meas,'padVelocity')
    pv=cur.meas.padVelocity(:);
    if numel(pv)>=2, padVel(1:2)=pv(1:2); end
end
if isfield(cur,'sem') && isfield(cur.sem,'energyMargin')
    margin=cur.sem.energyMargin;
end
end
