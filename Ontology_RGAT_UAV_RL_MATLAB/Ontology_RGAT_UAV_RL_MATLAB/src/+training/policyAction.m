function a = policyAction(policy,cur,env,cfg)
switch lower(policy.type)
    case 'expert'
        a=control.expertController(env.x,cfg);
    case 'expert_noisy'
        a=control.expertController(env.x,cfg);
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
