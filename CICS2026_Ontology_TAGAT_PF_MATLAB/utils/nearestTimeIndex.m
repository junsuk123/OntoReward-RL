function [idx,age] = nearestTimeIndex(tSrc,tq)
%NEARESTTIMEINDEX Nearest source sample for each query time.

idx = zeros(numel(tq),1);
age = inf(numel(tq),1);

j = 1;
for k = 1:numel(tq)
    while j < numel(tSrc) && abs(tSrc(j+1)-tq(k)) <= abs(tSrc(j)-tq(k))
        j = j+1;
    end
    idx(k) = j;
    age(k) = abs(tSrc(j)-tq(k));
end
end
