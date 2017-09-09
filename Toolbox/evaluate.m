function [ARI,AMI,NMI,ACC] = evaluate(C,numcomponents,gtlabels,nCluster)
addpath(genpath('../External/Measures'));

% evaluation using various clustering metrics
ARI = adjrand(C,gtlabels);
[AMI,NMI] = ami(gtlabels,C);
[ACC,~] = clustering_accuracy(gtlabels,C,numcomponents,nCluster);

% print result
fprintf('ARI = %0.3f |AMI = %0.3f |NMI = %0.3f |ACC = %0.3f |Numcomponents = %d \n',ARI,AMI,NMI,ACC,numcomponents)
end
