function P = toGPU(P,precision)
%TOGPU Move a parameter struct to the GPU at the requested precision.
%   Single by default: the model has a few thousand parameters and the loss is
%   an MSE on a tanh output, so double on the GPU buys nothing but bandwidth.
%   The result is gathered back to CPU double before it is saved, so a model
%   trained on the GPU is not tied to one.
if nargin<2 || isempty(precision), precision='single'; end
names=fieldnames(P);
for k=1:numel(names)
    v=P.(names{k});
    if isa(v,'dlarray'), v=extractdata(v); end
    P.(names{k})=dlarray(gpuArray(cast(v,precision)));
end
end
