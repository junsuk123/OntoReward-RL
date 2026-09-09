function [useGPU,why] = device(cfg,batchSize)
%DEVICE Decide whether an R-GAT batch of this size belongs on the GPU.
%   Measured on an RTX 4060 with the vectorized layer, per gradient step:
%
%       batch    CPU (double)    GPU (single)    GPU speedup
%          32        38 ms           48 ms           0.8x
%         256        47 ms           45 ms           1.0x
%        1024       131 ms           52 ms           2.5x
%        4096       481 ms           97 ms           4.9x
%
%   The layer is kernel-launch bound, not FLOP bound: the graph has 13 nodes and
%   a 24-wide hidden layer, so at the configured cfg.rgat.batchSize=32 the GPU
%   is slower than the CPU. Batch 256 is close enough to a tie to vary between
%   runs; 1024 is the first clear gain. Raising cfg.rgat.batchSize to get there
%   changes how many Adam steps the potential sees, so it is a hyperparameter
%   change and not a free speedup -- which is why 'auto' declines rather than
%   quietly rebatching.
%
%   matlab/tools/benchmark_rgat.m re-measures the crossover on this machine.
mode='auto'; minBatch=1024;
if isfield(cfg,'gpu')
    if isfield(cfg.gpu,'mode'), mode=lower(char(cfg.gpu.mode)); end
    if isfield(cfg.gpu,'minBatchForGPU'), minBatch=cfg.gpu.minBatchForGPU; end
end
switch mode
    case 'off'
        useGPU=false; why='cfg.gpu.mode is off'; return;
    case {'auto','on'}
        % fall through
    otherwise
        error('rgat:gpuMode','cfg.gpu.mode must be ''auto'', ''on'' or ''off''.');
end
if ~usableGPU()
    useGPU=false; why='no usable GPU'; return;
end
if strcmp(mode,'on')
    useGPU=true; why='cfg.gpu.mode is on'; return;
end
useGPU=batchSize>=minBatch;
if useGPU
    why=sprintf('batch %d >= cfg.gpu.minBatchForGPU %d',batchSize,minBatch);
else
    why=sprintf(['batch %d < cfg.gpu.minBatchForGPU %d; the GPU is slower than ' ...
        'the vectorized CPU path at this size'],batchSize,minBatch);
end
end

function tf = usableGPU()
tf=false;
if isempty(ver('parallel')), return; end
try
    tf=canUseGPU();
catch
    tf=false;
end
end
