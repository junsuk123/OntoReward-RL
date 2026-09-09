function C = make_ablation_configs()
% Returns four configs for the recommended ablation study.
base=default_config();

C=struct();

C.baseline=base; % Transformer handled separately

C.staticOntology=base;
C.staticOntology.graph.alphaOnt=1.0;
C.staticOntology.graph.alphaCov=0.0;
C.staticOntology.graph.alphaDyn=0.0;

C.dynamicNoOntology=base;
C.dynamicNoOntology.graph.alphaOnt=0.0;
C.dynamicNoOntology.graph.alphaCov=0.5;
C.dynamicNoOntology.graph.alphaDyn=0.5;

C.fullTAGAT=base;
end
