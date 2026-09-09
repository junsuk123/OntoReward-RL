function [p,m,v] = adamUpdateStruct(p,g,m,v,iter,lr,clipVal)
%ADAMUPDATESTRUCT Adam optimizer for a flat struct of dlarray parameters.

b1 = 0.9; b2 = 0.999; eps0 = 1e-8;
fields = fieldnames(p);

if isempty(m)
    for k=1:numel(fields)
        f=fields{k};
        m.(f)=dlarray(zeros(size(extractdata(p.(f))),"single"));
        v.(f)=dlarray(zeros(size(extractdata(p.(f))),"single"));
    end
end

for k=1:numel(fields)
    f = fields{k};
    grad = g.(f);
    if isempty(grad), continue; end
    grad = min(max(grad,-clipVal),clipVal);

    m.(f) = b1*m.(f) + (1-b1)*grad;
    v.(f) = b2*v.(f) + (1-b2)*(grad.^2);

    mh = m.(f)/(1-b1^iter);
    vh = v.(f)/(1-b2^iter);

    p.(f) = p.(f) - lr*mh./(sqrt(vh)+eps0);
end
end
