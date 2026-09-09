classdef SinusoidalPositionEncodingLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        Encoding
    end
    methods
        function layer=SinusoidalPositionEncodingLayer(dModel,maxLength,name)
            layer.Name=name;
            layer.Description="Fixed sinusoidal positional encoding";
            P=zeros(dModel,maxLength,'single');
            pos=single(0:maxLength-1);
            for i=1:2:dModel
                div=10000^((i-1)/dModel);
                P(i,:)=sin(pos/div);
                if i+1<=dModel, P(i+1,:)=cos(pos/div); end
            end
            layer.Encoding=P;
        end
        function Y=predict(layer,X)
            Y=doForward(layer,X);
        end
        function Y=forward(layer,X)
            Y=doForward(layer,X);
        end
    end
    methods (Access=private)
        function Y=doForward(layer,X)
            fmt=dims(X); Xu=stripdims(X); T=size(Xu,3);
            P=reshape(layer.Encoding(:,1:T),size(layer.Encoding,1),1,T);
            Yu=Xu+P;
            Y=dlarray(Yu,fmt);
        end
    end
end
