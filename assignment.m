function [C,numcomponents] = assignment(U,i,j,epsilon,nsamples)
% NOTE: We use external gptoolbox package to compute connected components. 
% MATLAB >= v2016a includes inbuilt conncomp function.
% One can accordingly modify the code to use that.
addpath(genpath('External/gptoolbox-master/'))

diff = sum((U(i,:) - U(j,:)).^2,2);

% computing connected components. 
isConn = sqrt(diff) < epsilon;
G = sparse([i(isConn);j(isConn)], [j(isConn);i(isConn)], ones(2*sum(isConn),1),nsamples,nsamples);
[numcomponents,C] = conncomp(G);
end
