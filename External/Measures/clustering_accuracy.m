function [acc,cluster_map] = clustering_accuracy(gtlabels,estlabels,numcomponents,nCluster)
cost_matrix = zeros(numcomponents,nCluster);
categories = unique(gtlabels);
for i = 1:nCluster
    [cost_matrix(:,i),~] = histcounts(estlabels(gtlabels==categories(i)), 1:numcomponents+1);
end
[col_ind,~] = munkres(max(max(cost_matrix)) - cost_matrix);
cluster_map = col_ind;
cost_matrix(col_ind == 0,:) = [];
col_ind(col_ind==0) = [];
acc = sum(diag(cost_matrix(:,col_ind)));

acc = acc / length(gtlabels);
end