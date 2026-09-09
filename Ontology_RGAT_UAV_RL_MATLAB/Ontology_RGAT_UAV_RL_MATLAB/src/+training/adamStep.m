function [P,opt] = adamStep(P,G,opt,lr,t,clipNorm)
%ADAMSTEP Adam update for a flat struct of dlarray parameters.
if nargin<6 || isempty(clipNorm), clipNorm=inf; end
names=fieldnames(P);
% Global norm clipping.
sq=0;
for i=1:numel(names)
    g=G.(names{i});
    if ~isempty(g), sq=sq+sum(extractdata(g).^2,'all'); end
end
gn=sqrt(sq); scale=min(1,clipNorm/max(gn,1e-12));
for i=1:numel(names)
    n=names{i}; g=G.(n)*scale;
    if ~isfield(opt,'m') || ~isfield(opt.m,n)
        opt.m.(n)=P.(n)*0; opt.v.(n)=P.(n)*0;
    end
    opt.m.(n)=0.9*opt.m.(n)+0.1*g;
    opt.v.(n)=0.999*opt.v.(n)+0.001*(g.^2);
    mh=opt.m.(n)/(1-0.9^t); vh=opt.v.(n)/(1-0.999^t);
    P.(n)=P.(n)-lr*mh./(sqrt(vh)+1e-8);
end
end
