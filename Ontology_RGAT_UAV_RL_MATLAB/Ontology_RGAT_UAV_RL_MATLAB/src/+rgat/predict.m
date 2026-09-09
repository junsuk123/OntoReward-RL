function phi = predict(P,graph)
y=rgat.forward(P,graph.X,graph);
phi=double(extractdata(y));
end
