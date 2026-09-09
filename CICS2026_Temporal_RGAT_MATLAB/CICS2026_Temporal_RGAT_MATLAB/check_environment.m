function check_environment()
%CHECK_ENVIRONMENT Verify required MATLAB environment.
fprintf('\n=== CICS2026 environment check ===\n');
fprintf('MATLAB %s | release %s\n',version,version('-release'));
info=ver('nnet');
if isempty(info)
    error('Deep Learning Toolbox is required.');
end
if ~license('test','Neural_Network_Toolbox')
    error('Deep Learning Toolbox license is not available.');
end
req={"dlnetwork","selfAttentionLayer","globalAveragePooling1dLayer", ...
     "layerNormalizationLayer","geluLayer","adamupdate","dlfeval","dlgradient"};
for i=1:numel(req)
    % dlgradient is a dlarray method, so exist() returns 0 for it; which() resolves it.
    ok=exist(req{i},'file')==2 || exist(req{i},'class')==8 || ~isempty(which(req{i}));
    fprintf('[%s] %s\n', ternary(ok,' OK ','FAIL'), req{i});
    if ~ok, error('Required function/layer missing: %s',req{i}); end
end

% Verify that the EDL KL operations support autodiff in this MATLAB release.
a=dlarray(single([2.0;3.0]));
try
    [~,g]=dlfeval(@kl_test,a);
    if any(~isfinite(extractdata(g)),'all')
        error('non-finite gradient');
    end
    fprintf('[ OK ] Dirichlet KL autodiff (gammaln/psi)\n');
catch ME
    error(['EDL KL autodiff test failed. This package targets MATLAB R2025b. ' ...
        'Error: ' ME.message]);
end
fprintf('Environment ready.\n\n');
end

function [L,g]=kl_test(a)
alpha=softplus_stable(a)+1;
L=dirichlet_kl_uniform(alpha);
g=dlgradient(L,a);
end

function out=ternary(c,a,b)
if c, out=a; else, out=b; end
end
