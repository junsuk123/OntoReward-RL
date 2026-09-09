function [pFault,uncertainty,details]=edl_decode(raw,mode)
%EDL_DECODE Convert network raw output to probability/uncertainty.
raw=strip_if_needed(raw); mode=string(mode);
switch mode
    case "softmax_hard"
        P=softmax_cols(raw); pFault=P(2,:); uncertainty=nan(size(pFault),'single'); details=struct();
    case {"edl_hard","edl_soft"}
        a=softplus_stable(raw)+1; S=sum(a,1); pFault=a(2,:)./S; uncertainty=2./S; details=struct('alpha',a);
    case "dual_edl"
        a1=softplus_stable(raw(1:2,:))+1; a2=softplus_stable(raw(3:4,:))+1;
        u1=2./sum(a1,1); u2=2./sum(a2,1);
        af=(1-u1).*(a1-1)+(1-u2).*(a2-1)+1;
        pFault=af(2,:)./sum(af,1); uncertainty=2./sum(af,1);
        details=struct('alpha1',a1,'alpha2',a2,'alphaFused',af,'u1',u1,'u2',u2);
    otherwise
        error('Unknown decode mode %s',mode);
end
end
function X=strip_if_needed(X)
if isa(X,'dlarray'), X=stripdims(X); end
end
function P=softmax_cols(Z)
Z=Z-max(Z,[],1); E=exp(Z); P=E./(sum(E,1)+1e-8);
end
