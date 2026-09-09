function generate_demo_datasets()
%GENERATE_DEMO_DATASETS Generate deterministic synthetic smoke-test data.
% Counts follow the presentation diagnostics only. These are NOT research data.
cfg=cics_config("demo");
rng(cfg.seed,'twister');
if ~isfolder(cfg.dataDir), mkdir(cfg.dataDir); end
make_one('medium',658,206,0.75,cfg.files.medium);
make_one('harsh',2314,1569,1.15,cfg.files.harsh);
make_one('deep',1539,521,1.35,cfg.files.deep);
fprintf('Synthetic demo CSVs generated in %s\n',cfg.dataDir);
end

function make_one(name,n,nFault,difficulty,outfile)
% Smooth latent severity -> contiguous-ish fault intervals.
z=filter(1,[1 -0.96],randn(n,1));
z=(z-min(z))/(max(z)-min(z)+eps);
[~,ord]=sort(z,'descend');
y=zeros(n,1); y(ord(1:nFault))=1;
% Smooth label neighborhood while preserving exact hard count via severity ranking.
sev=0.25*z + 0.75*movmean(single(y),9);
sev=(sev-min(sev))/(max(sev)-min(sev)+eps);

noise=@(s) s*randn(n,1);
numSV = max(5,round(29 - (8*difficulty).*sev + noise(1.3)));
hDOP = max(0.3,0.70 + 1.10*difficulty*sev + abs(noise(0.15)));
vDOP = max(0.5,0.95 + 1.35*difficulty*sev + abs(noise(0.20)));
hAcc = max(0.2,1.4 + 2.8*difficulty*sev + abs(noise(0.35)));
vAcc = max(0.3,2.0 + 3.7*difficulty*sev + abs(noise(0.45)));
gSpeed = max(0,7 + 2*sin((1:n)'/43) + noise(0.7));
CNO_mean = 31 - 9*difficulty*sev + noise(1.5);
CNO_std = max(0.1,4.5 + 4.0*difficulty*sev + abs(noise(0.9)));
CNO_gap = 2 + 12*difficulty*sev + noise(2.2);
low_elev_ratio = min(0.9,max(0.02,0.08 + 0.25*difficulty*sev + noise(0.025)));
PR_RMS = max(0,6 + 35*difficulty*sev + abs(noise(5)));
Fault_SVID_count = max(0,round(0.2 + 3.8*difficulty*sev + noise(0.7)));

% Provenance-like columns. Hard label is exactly threshold-consistent.
pos_error_2d = (1-y).*(1.2 + abs(noise(0.55))) + y.*(3.5 + 4.5*sev + abs(noise(0.8)));
pos_error_2d(y==0)=min(pos_error_2d(y==0),2.95);
pos_error_2d(y==1)=max(pos_error_2d(y==1),3.05);
residual = 0.3 + 3.2*difficulty*sev + abs(noise(0.45));
soft_fault_prob = 1./(1+exp(-(1.5*residual - 2.6))) + noise(0.035);
soft_fault_prob = min(1,max(0,soft_fault_prob));

timestamp=(0:n-1)';
scenario=repmat(string(name),n,1);
hard_label=y;
T=table(timestamp,numSV,hDOP,vDOP,hAcc,vAcc,gSpeed,CNO_mean,CNO_std,CNO_gap, ...
    low_elev_ratio,PR_RMS,Fault_SVID_count,hard_label,soft_fault_prob,pos_error_2d,residual,scenario);
writetable(T,outfile);
end
