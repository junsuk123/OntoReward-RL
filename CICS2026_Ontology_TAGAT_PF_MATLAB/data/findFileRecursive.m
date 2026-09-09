function pathOut = findFileRecursive(rootDir,fileName)
%FINDFILERECURSIVE Return first exact filename match under rootDir.

d = dir(fullfile(rootDir,"**",fileName));
d = d(~[d.isdir]);
if isempty(d)
    pathOut = "";
else
    pathOut = string(fullfile(d(1).folder,d(1).name));
end
end
