% wmon_episodes.m -- episodes of run_WMON_demo.m's simulateEpisode (verbatim local functions
% below, lines 246-394 of the 2026-11-10 script) for tools/wmon_vs_kalman_benchmark.py.
%   cd <output dir>; matlab -batch wmon_episodes   (needs WMON_output/WMON_result_data.mat)
% Episodes of run_WMON_demo.m's simulateEpisode (verbatim copy below) for the
% WMON vs Kalman comparison: train 1..12, validation 501..503, MATLAB held-out
% 504..507, extended held-out 508..557, plus MATLAB WMON output per episode.
cfg.dt = 0.1; cfg.T = 90; cfg.win = 12; cfg.nModes = 3;
d = load('/home/j/Downloads/WMON_output/WMON_result_data.mat');
seeds = [1:12, 501:557];
isTest = seeds > 500;
E = numel(seeds);
obs = zeros(E, 6, cfg.T); xTrue = obs; wmonState = obs; posValid = zeros(E, cfg.T);
for i = 1:E
    ep = simulateEpisode(cfg, seeds(i), isTest(i));
    obs(i,:,:) = ep.obs; xTrue(i,:,:) = ep.xTrue; posValid(i,:) = ep.posValid;
    est = inferEpisode(d.theta, ep, cfg, true);
    wmonState(i,:,:) = est.state;
end
theta = d.theta;
save('wmon_episodes.mat', 'seeds', 'isTest', 'obs', 'xTrue', 'posValid', 'wmonState', 'theta', '-v7');
fprintf('saved %d episodes\n', E);
function ep = simulateEpisode(cfg,seed,isTest)
    % Distinct train/test motion patterns and sensor perturbations.
    rng(seed,'twister');
    T = cfg.T; dt = cfg.dt; t = (0:T-1)*dt;
    x = zeros(6,T); % [px py vx vy ax ay]'
    x(1:2,1) = [2*randn; 2*randn];
    x(3:4,1) = [0.8*randn;0.7*randn];
    phase = 2*pi*rand(2,1);
    gain = (0.34+0.12*rand(2,1))*(1+0.25*isTest);
    turnTime = 3.0+2*rand;
    for k=1:T
        tt = t(k);
        accel = gain.*sin([0.65;0.42]*tt + phase);
        if tt>turnTime && tt<turnTime+2.1
            accel = accel + [0.28;-0.22]*(1+0.45*isTest);
        end
        x(5:6,k) = accel;
        if k<T
            x(1:2,k+1) = x(1:2,k)+dt*x(3:4,k)+0.5*dt^2*accel;
            x(3:4,k+1) = x(3:4,k)+dt*accel;
        end
    end
    posNoise = 0.26*(1+0.2*isTest);
    velNoise = 0.16*(1+0.15*isTest);
    accNoise = 0.19*(1+0.2*isTest);
    obs = x + [posNoise*randn(2,T); velNoise*randn(2,T); accNoise*randn(2,T)];
    posValid = true(1,T);
    % Intermittent position sensor dropout; different time interval in test.
    dropoutStart = 38 + randi(13) + 4*isTest;
    dropoutLen = 12+randi(9)+4*isTest;
    posValid(dropoutStart:min(T,dropoutStart+dropoutLen)) = false;
    posValid(rand(1,T)<0.035) = false;
    posValid(1:cfg.win)=true; % stable initialization
    % The missing position is NaN, never accessed as a current measurement.
    obs(1:2,~posValid) = NaN;
    ep.t=t; ep.xTrue=x; ep.obs=obs; ep.posValid=posValid;
end

function val = batchLoss(theta,episodes,cfg)
    val = 0;
    for e=1:numel(episodes)
        ep=episodes{e};
        estimated=inferEpisode(theta,ep,cfg,true);
        ids=cfg.win:cfg.T;
        pErr=estimated.state(1:2,ids)-ep.xTrue(1:2,ids);
        vErr=estimated.state(3:4,ids)-ep.xTrue(3:4,ids);
        val=val+mean(pErr(:).^2)+0.35*mean(vErr(:).^2);
    end
    val=val/numel(episodes);
end

function est = inferEpisode(theta,ep,cfg,useOntology)
    T=cfg.T; dt=cfg.dt;
    state=zeros(6,T);
    energy=zeros(T,5);
    confidence=zeros(T,3);
    debug=cell(1,T);
    data=fillMissingPosition(ep.obs,ep.posValid);
    state(:,1)=[data(1:4,1);data(5:6,1)];
    weights=exp(theta(1:5));
    mix=sigmoid(theta(6));
    noiseSensitivity=exp(theta(7));
    for k=1:T
        % L0: wave field for the three semantic modalities.
        windowIndices=max(1,k-cfg.win+1):k;
        segment=data(:,windowIndices);
        if size(segment,2)<cfg.win
            segment=[repmat(segment(:,1),1,cfg.win-size(segment,2)),segment];
        end
        freq=fft(segment,[],2)/cfg.win;
        mask=zeros(1,cfg.win);
        mask(1:cfg.nModes+1)=1;
        mask(end-cfg.nModes+1:end)=1;
        filtered=real(ifft(freq.*repmat(mask,6,1),[],2)*cfg.win);
        sample=(1-mix)*segment(:,end)+mix*filtered(:,end);

        % L1-L2: information particles = modal energy + reliability.
        bandPower=abs(freq).^2;
        highBand=bandPower.*repmat(1-mask,6,1);
        totalPower=sum(bandPower(:,2:end),2)+1e-4;
        hfRatio=sum(highBand,2)./totalPower;
        particleConf=max(0.15,1./(1+noiseSensitivity*hfRatio));
        particleConf(1:2)=particleConf(1:2)*double(ep.posValid(k));
        confidence(k,:)=mean(reshape(particleConf,2,3),1);

        % L3: three semantic atoms per axis: Position, Velocity, Accel.
        atomState=zeros(2,3);
        if k==1
            state(:,k)=sample;
            debug{k}=debugInfo(segment,filtered,particleConf,atomState);
            continue
        end
        for d=1:2
            ids=[d,d+2,d+4];
            q=[sample(ids(1));sample(ids(2));sample(ids(3))];
            wp=weights(1)*particleConf(ids(1));
            wv=weights(2)*particleConf(ids(2));
            wa=weights(3)*particleConf(ids(3));
            if ~ep.posValid(k), wp=0; end
            if useOntology
                kp=weights(4); kv=weights(5);
            else
                kp=0; kv=0;
            end
            prevP=state(d,k-1); prevV=state(d+2,k-1);
            % L4: minimize sum of semantic on-site & bond potentials.
            % Residuals: p-p_prev-dt*(v+v_prev)/2 and v-v_prev-dt*a.
            A=[1 0 0;0 1 0;0 0 1;1 -dt/2 0;0 1 -dt];
            b=[q(1);q(2);q(3);prevP+dt/2*prevV;prevV];
            W=[wp;wv;wa;kp;kv];
            H=A'*(A.*W)+1e-8*eye(3);
            rhs=A'*(W.*b);
            z=H\rhs;
            state(ids,k)=z;
            atomState(d,:)=z';
            energy(k,:)=energy(k,:)+0.5*(W'.*((A*z-b)').^2);
        end
        debug{k}=debugInfo(segment,filtered,particleConf,atomState);
    end
    est.state=state;
    est.energy=energy;
    est.confidence=confidence;
    est.debug=debug;
end

function out=debugInfo(segment,filtered,particleConf,atomState)
    out.rawWindow=segment;
    out.reconWindow=filtered;
    out.particleConfidence=reshape(particleConf,2,3);
    out.atomState=atomState;
end

function data=fillMissingPosition(obs,posValid)
    % Causal zero-order hold solely for FFT window preparation.
    data=obs;
    for k=2:size(data,2)
        if ~posValid(k)
            data(1:2,k)=data(1:2,k-1);
        end
    end
end

function out=rawHoldEstimate(ep)
    out.state=fillMissingPosition(ep.obs,ep.posValid);
end

function y=sigmoid(x)
    y=1./(1+exp(-x));
end
