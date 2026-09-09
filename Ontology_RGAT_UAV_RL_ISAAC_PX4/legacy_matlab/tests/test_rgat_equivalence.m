function tests = test_rgat_equivalence
%TEST_RGAT_EQUIVALENCE Vectorized overlay against the original edge loop.
tests=functiontests(localfunctions);
end

function testSingleAndBatch(testCase)
here=fileparts(mfilename('fullpath'));
cd(fullfile(here,'..'));
setup_external_path();
cfg=defaultExternalConfig('quick');
sem=struct('positionError',0.7,'verticalSpeed',-0.4,'tilt',0.1, ...
    'angularRate',0.2,'windRisk',0.3,'markerQuality',0.8, ...
    'visualStability',0.8,'alignment',0.4,'attitudeStability',0.6, ...
    'touchdownSafety',0.2,'padMotion',0.35,'batteryReserve',0.6);
graph=semantic.buildOntologyGraph(sem,cfg);
model=rgat.unwrap(rgat.initModel(cfg));
X=repmat(graph.X,1,1,8)+0.01*randn(cfg.ontology.inDim,cfg.ontology.nNodes,8);

expected=referenceForward(model,X,graph);
actual=double(rgat.forward(model,X,graph));
verifyEqual(testCase,actual,expected,'AbsTol',1e-12);
verifyEqual(testCase,double(rgat.forward(model,X(:,:,3),graph)), ...
    expected(3),'AbsTol',1e-12);
end

function phi = referenceForward(P,X,graph)
B=size(X,3); phi=zeros(1,B);
for b=1:B
    H1=tanh(referenceLayer(X(:,:,b),P.W1,P.a1,P.E1,graph));
    H2=tanh(referenceLayer(H1,P.W2,P.a2,P.E2,graph)+H1);
    phi(b)=tanh(P.wOut*H2(:,graph.goalNode)+P.bOut);
end
end

function Z = referenceLayer(H,W,a,E,graph)
N=size(H,2); columns=cell(1,N);
for j=1:N
    incoming=find(graph.dst==j);
    scores=zeros(1,numel(incoming)); messages=zeros(size(W,1),numel(incoming));
    for k=1:numel(incoming)
        edge=incoming(k); source=graph.src(edge); relation=graph.rel(edge);
        hs=W(:,:,relation)*H(:,source);
        hd=W(:,:,relation)*H(:,j);
        raw=a(:,:,relation)*[hs;hd;E(:,relation)];
        scores(k)=0.6*raw+0.4*abs(raw);
        messages(:,k)=hs;
    end
    weights=exp(scores); weights=weights/(sum(weights)+1e-9);
    columns{j}=sum(messages.*weights,2);
end
Z=cat(2,columns{:});
end
