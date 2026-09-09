function p = weak_label_surrogate(T,cfg,scenario)
%WEAK_LABEL_SURROGATE Build the non-strict weak label from a stated source.
%
% Neither option is the prior study's estimator. They fail in opposite
% directions and the choice must be stated in any write-up:
%
%   "pr_rms"    the surrogate shipped with the public reconstruction,
%               logistic(PR_RMS/faultResidualM - 1). Two separate defects:
%               (1) it is degenerate - PR_RMS runs ~46 against a centre of 10,
%                   so the logistic saturates at mean 0.89, sd 0.16, and the
%                   weak head is handed a near-constant target;
%               (2) rescaling does not save it - PR_RMS is rank-uninformative
%                   about the fault label (AUROC 0.471, i.e. below chance), so
%                   no monotone transform of it can carry signal.
%               Honest, and the honest answer is that the weak head has nothing
%               to learn. The dual fusion is then strictly worse than the hard
%               head alone, which is a real result about this reconstruction -
%               not a result about the architecture.
%
%   "pos_error" logistic((pos_error_2d - hardFaultThresholdM)/tau), a smooth
%               version of the very threshold that defines hard_label. This
%               matches what the prior study evidently did - their SoftLabel
%               baseline, which uses the soft label directly as the prediction,
%               scored F1 0.959 - but it is CIRCULAR here: it recovers AUROC
%               1.000 by construction. Use it to exercise and compare the dual
%               architecture, never to claim the weak head adds information.
src=string(cfg.weakLabel.source);
switch src
    case "pr_rms"
        if ~ismember("soft_fault_prob_surrogate",string(T.Properties.VariableNames))
            error('%s has no soft_fault_prob_surrogate column.',scenario);
        end
        p=T.soft_fault_prob_surrogate;
    case "pos_error"
        if ~ismember("pos_error_2d",string(T.Properties.VariableNames))
            error('%s has no pos_error_2d column for the pos_error weak label.',scenario);
        end
        tau=max(cfg.weakLabel.errorTauM,eps);
        p=1./(1+exp(-(T.pos_error_2d-cfg.hardFaultThresholdM)/tau));
        warning('cics:circularWeakLabel', ...
            ['%s: weak label derived from pos_error_2d, the same quantity that defines ' ...
             'hard_label. Dual-EDL gains under this setting are circular and must not be ' ...
             'reported as evidence that the weak head adds information.'],scenario);
    otherwise
        error('Unknown cfg.weakLabel.source "%s".',src);
end
p=double(p); p=min(max(p,0),1);
end
