function results = benchmark_rgat(batchSizes,repetitions)
%BENCHMARK_RGAT Measure the CPU/GPU crossover for the active R-GAT overlay.
%   RESULTS=BENCHMARK_RGAT() times one forward/backward pass at representative
%   batch sizes. It does not modify training hyperparameters or saved models.
if nargin<1 || isempty(batchSizes), batchSizes=[32 256 1024 4096]; end
if nargin<2 || isempty(repetitions), repetitions=5; end

here=fileparts(mfilename('fullpath'));
cd(fullfile(here,'..'));
setup_external_path();
cfg=defaultExternalConfig('quick'); cfg.viz.training=false;
sem=struct('positionError',0.7,'verticalSpeed',-0.4,'tilt',0.1, ...
    'angularRate',0.2,'windRisk',0.3,'markerQuality',0.8, ...
    'visualStability',0.8,'alignment',0.4,'attitudeStability',0.6, ...
    'touchdownSafety',0.2,'padMotion',0.35,'batteryReserve',0.6);
graph=semantic.buildOntologyGraph(sem,cfg);
cpuModel=rgat.initModel(cfg);
gpuAvailable=canBenchmarkGPU();
if gpuAvailable, gpuModel=rgat.toGPU(cpuModel,cfg.gpu.precision); end

n=numel(batchSizes); BatchSize=batchSizes(:);
CPUms=zeros(n,1); GPUms=nan(n,1); GPUSpeedup=nan(n,1);
for row=1:n
    B=BatchSize(row);
    X=repmat(graph.X,1,1,B)+0.01*randn(cfg.ontology.inDim,cfg.ontology.nNodes,B);
    y=0.5*randn(1,B);
    dlfeval(@training.rgatGradients,cpuModel,X,y,graph);
    started=tic;
    for k=1:repetitions
        dlfeval(@training.rgatGradients,cpuModel,X,y,graph);
    end
    CPUms(row)=1e3*toc(started)/repetitions;

    if gpuAvailable
        Xgpu=gpuArray(cast(X,cfg.gpu.precision));
        ygpu=gpuArray(cast(y,cfg.gpu.precision));
        dlfeval(@training.rgatGradients,gpuModel,Xgpu,ygpu,graph);
        wait(gpuDevice); started=tic;
        for k=1:repetitions
            dlfeval(@training.rgatGradients,gpuModel,Xgpu,ygpu,graph);
        end
        wait(gpuDevice); GPUms(row)=1e3*toc(started)/repetitions;
        GPUSpeedup(row)=CPUms(row)/GPUms(row);
    end
end
results=table(BatchSize,CPUms,GPUms,GPUSpeedup);
disp(results);
end

function tf = canBenchmarkGPU()
tf=false;
if isempty(ver('parallel')), return; end
try
    tf=canUseGPU();
catch
end
end
