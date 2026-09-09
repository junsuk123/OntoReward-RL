function saveConfig(cfg,filePath)
try
    txt=jsonencode(cfg,"PrettyPrint",true);
catch
    txt=jsonencode(cfg);
end
fid=fopen(filePath,"w");
if fid<0,error("Cannot write config: %s",filePath);end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid,txt,"char");
end
