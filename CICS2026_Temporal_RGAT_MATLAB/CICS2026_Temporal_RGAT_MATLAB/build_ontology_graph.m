function G = build_ontology_graph(cfg,collapseRelations)
%BUILD_ONTOLOGY_GRAPH Typed adjacency over the 12 GNSS feature nodes.
% Rows=target nodes, columns=source nodes.
if nargin<2, collapseRelations=false; end
N=numel(cfg.featureNames);
relations=["geometry","accuracy","signal","consistency","motion","self"];
R=numel(relations); A=zeros(N,N,R,'single');
idx=@(name)find(cfg.featureNames==name,1);

% geometry
add("hDOP","numSV",1); add("vDOP","numSV",1);
add("hDOP","low_elev_ratio",1); add("vDOP","low_elev_ratio",1);
% accuracy
add("hAcc","hDOP",2); add("vAcc","vDOP",2);
add("hAcc","PR_RMS",2); add("vAcc","PR_RMS",2);
% signal
add("CNO_std","CNO_mean",3); add("CNO_gap","CNO_mean",3);
add("CNO_gap","low_elev_ratio",3); add("Fault_SVID_count","CNO_gap",3);
% consistency / degradation
add("PR_RMS","CNO_gap",4); add("PR_RMS","CNO_std",4);
add("hAcc","Fault_SVID_count",4); add("vAcc","Fault_SVID_count",4);
% motion context
add("hAcc","gSpeed",5); add("PR_RMS","gSpeed",5);
% self loops
for i=1:N, A(i,i,6)=1; end
% Reverse semantic edges for information flow robustness (same relation type).
for r=1:5, A(:,:,r)=max(A(:,:,r),A(:,:,r).'); end

if collapseRelations
    U=max(A,[],3);
    A=zeros(N,N,R,'single');
    A(:,:,1)=U; A(:,:,6)=eye(N,'single');
end
G=struct('adjacency',A,'relations',relations,'featureNames',cfg.featureNames);

    function add(target,source,r)
        A(idx(target),idx(source),r)=1;
    end
end
