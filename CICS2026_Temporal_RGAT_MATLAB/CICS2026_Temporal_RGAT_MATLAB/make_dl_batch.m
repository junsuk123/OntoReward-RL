function dlX=make_dl_batch(Xb)
%MAKE_DL_BATCH Create CBT dlarray and use GPU only when available.
Xb=single(Xb);
if gpu_is_usable(), Xb=gpuArray(Xb); end
dlX=dlarray(Xb,'CBT');
end

function tf=gpu_is_usable()
%GPU_IS_USABLE Probe once per session, not once per minibatch.
% gpuDeviceCount("available") excludes devices this release's CUDA libraries
% cannot drive, which is what happens on a GPU newer than the MATLAB release.
persistent cached
if isempty(cached)
    cached=false;
    if exist('gpuDeviceCount','file')==2
        try
            cached=gpuDeviceCount("available")>0;
        catch
            try
                cached=gpuDeviceCount>0;
            catch
                cached=false;
            end
        end
    end
end
tf=cached;
end
