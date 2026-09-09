function a = policyAction(policy,cur,env,cfg)
%POLICYACTION External version: the expert also sees the semantic context.
%   The behavior policy has to know what the deck is doing and how much energy
%   is left, so it receives CUR rather than the raw state alone. The original
%   signature took only env.x, which is exactly the information that is no
%   longer sufficient.
switch lower(policy.type)
    case 'expert'
        a=control.expertController(env.x,cfg,cur);
    case 'expert_noisy'
        a=control.expertController(env.x,cfg,cur);
        ns=policy.noiseStd;
        a=a+ns*randn(4,1);
        % Occasionally inject a larger perturbation to create failure examples.
        if rand < min(0.20,2*ns)
            a=a+[0.2;0.5;0.5;0.3].*randn(4,1);
        end
        a=max(-1,min(1,a));
    case 'ppo'
        a=training.samplePolicy(policy.agent.actor,cur.obs,logical(policy.deterministic),cfg);
    otherwise
        error('Unknown policy type: %s',policy.type);
end
end
