function describe_splits(SP)
%DESCRIBE_SPLITS Print split sizes and class balance so the protocol is visible.
fprintf('\nSplit protocol: %s\n',SP.mode);
fprintf('  %-6s %7s %12s\n','set','windows','fault rate');
fprintf('  %-6s %7d %12.3f\n','train',numel(SP.YhTrain),mean(SP.YhTrain));
if ~isempty(SP.YhVal)
    fprintf('  %-6s %7d %12.3f\n','val',numel(SP.YhVal),mean(SP.YhVal));
end
fprintf('  %-6s %7d %12.3f\n','test',numel(SP.YhTest),mean(SP.YhTest));
end
