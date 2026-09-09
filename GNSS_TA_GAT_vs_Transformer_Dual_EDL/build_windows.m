function [X,Yhard,Ysoft,meta] = build_windows(seq,cfg)
% X: [C x W x B]
Xcell = {};
Yhard = [];
Ysoft = [];
env = strings(0,1);
endIndex = [];

for s = 1:numel(seq)
    D = seq{s};
    n = D.n;
    for t = cfg.window:cfg.stride:n
        w = D.X(t-cfg.window+1:t,:).'; % [C x W]
        Xcell{end+1} = single(w); %#ok<AGROW>
        Yhard(end+1) = single(D.yHard(t)); %#ok<AGROW>
        Ysoft(end+1) = single(D.ySoft(t)); %#ok<AGROW>
        env(end+1,1) = string(D.env); %#ok<AGROW>
        endIndex(end+1,1) = t; %#ok<AGROW>
    end
end

X = cat(3,Xcell{:});
Yhard = single(Yhard(:).');
Ysoft = single(Ysoft(:).');

meta = table(env,endIndex,'VariableNames',{'Environment','EndIndex'});
end
