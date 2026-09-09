function T = summarizeDataset(data)
%SUMMARIZEDATASET Compact per-feature diagnostics for reproducibility.

F = data.features;
vars = string(F.Properties.VariableNames);
n = numel(vars);
Mean = zeros(n,1);
Std = zeros(n,1);
Min = zeros(n,1);
Max = zeros(n,1);
MissingPct = zeros(n,1);

for k=1:n
    x = F.(vars(k));
    Mean(k)=mean(x,"omitnan");
    Std(k)=std(x,0,"omitnan");
    Min(k)=min(x,[],"omitnan");
    Max(k)=max(x,[],"omitnan");
    MissingPct(k)=100*mean(~isfinite(x));
end

T=table(vars.',Mean,Std,Min,Max,MissingPct, ...
    'VariableNames',["Feature","Mean","Std","Min","Max","MissingPct"]);
end
