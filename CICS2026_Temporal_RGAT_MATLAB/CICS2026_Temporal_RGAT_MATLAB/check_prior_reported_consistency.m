function T=check_prior_reported_consistency()
%CHECK_PRIOR_REPORTED_CONSISTENCY Preserve slide values and flag arithmetic mismatch.
root=fileparts(mfilename('fullpath'));
T=readtable(fullfile(root,'prior_reported_metrics.csv'));
T.F1FromReportedPR=2*T.ReportedPrecision.*T.ReportedRecall./max(eps,T.ReportedPrecision+T.ReportedRecall);
T.AbsDifference=abs(T.ReportedF1-T.F1FromReportedPR);
T.ArithmeticConsistent=T.AbsDifference<0.005;
disp(T);
if any(~T.ArithmeticConsistent)
    warning(['The presentation contains reported F1 values that are not arithmetically ' ...
        'consistent with the displayed Precision/Recall for some rows. The package preserves ' ...
        'the slide values and reports this diagnostic instead of silently changing them.']);
end
end
