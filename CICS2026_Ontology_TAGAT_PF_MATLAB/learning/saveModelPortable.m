function saveModelPortable(model,filePath)
%SAVEMODELPORTABLE Save model with numeric arrays instead of live dlarray objects.

portable = model;
fn = fieldnames(model.params);
for k=1:numel(fn)
    f=fn{k};
    portable.params.(f)=extractdata(model.params.(f));
end
save(filePath,"portable","-v7.3");
end
