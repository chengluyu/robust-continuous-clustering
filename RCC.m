function [clustAssign,numcomponents,optTime,gtlabels,nCluster] = RCC(filename, maxiter, inner_iter)
% set display variable for printing intermediate objective
disp = 1; % 0 or 1

%% Load data and convert them to double precision
load(filename);

X = double(X); % features stacked as N x D (D is the dimension)
w = double(w); % list of edges represented by start and end nodes

% The node indices should starts with 1
if(min(gtlabels) == 0)
    gtlabels = double(gtlabels) + 1;
end
if(min(w(:)) == 0)
    i = w(:,1)+1;
    j = w(:,2)+1;
end

%% initialization
[nsamples, nfeatures] = size(X);
nCluster = length(unique(gtlabels));
npairs = size(w,1);

% precomputing \xi
xi = norm(X,2);

% setting weights as given in equation [S1]
R = sparse([i;j], [j;i], [ones(npairs,1);ones(npairs,1)], nsamples, nsamples);
nconn = full(sum(R,2));
weights = mean(nconn) ./ sqrt(nconn(i).*nconn(j));

% initializing U to X and lpq = 1 \forall p,q \in E
U = X; 
lpq = ones(length(i),1);

% computation of \delta, \mu
epsilon = sqrt(sum((X(i,:) - X(j,:)).^2,2));
% for epsilon computation:- not considering the edges with both the nodes 
% being near duplicate of each other.
epsilon(epsilon/sqrt(nfeatures) < 1e-2) = [];
[epsilon,~] = sort(epsilon);

mu = 3 * epsilon(end)^2; % setting \mu to be 3r^2

robsamp = min(250, ceil(npairs*0.01)); % top 1% of the closest neighbours
delta = mean(epsilon(1:robsamp));
epsilon = mean(epsilon(1:ceil(npairs*0.01)));

% computation of matrix A = D-R (here D is the diagonal matrix and R is
% symmetric matrix)
R = sparse([i;j], [j;i], [lpq.*weights;lpq.*weights], nsamples, nsamples);
D = sparse(1:nsamples, 1:nsamples, sum(R,2), nsamples, nsamples);

% initial computation of \lambda
lambda =  xi / (eigs(D-R,1));

count = 0;
if(disp)
    fprintf('mu = %f, lambda = %f, epsilon = %f, delta = %f \n',mu, lambda, epsilon, delta)    
    fprintf(' Iter | Data \t | Smooth \t | Obj \t | Time(s) \n')    
end
obj = zeros(maxiter+1,1);

%% start of optimization phase
starttime = tic;

for iter = 2:maxiter+1
    
    % update lpq
    lpq = GemanMcClure(U(i,:)-U(j,:),mu);
    
    % compute objective
    [obj(iter)] = computeObj(X,U,lpq,i,j,lambda,mu,weights,disp,iter-1,starttime);
    
    % update U's
    R = sparse([i;j], [j;i], [weights.*lpq;weights.*lpq], nsamples, nsamples);
    D = sparse(1:nsamples, 1:nsamples, sum(R,2), nsamples, nsamples);
    M = speye(nsamples) + lambda * (D-R);
        pfun = cmg_sdd(M);
    % For large number of features, computation can be easily set in parallel
    parfor k = 1:nfeatures
        [U(:,k),~] = pcg(M,X(:,k),1e-2,100,pfun);
    end
    
    % check for stopping criteria
    count = count + 1;
    if (abs(obj(iter-1)-obj(iter)) < 1e-1) || count == inner_iter
        if mu >= delta
            mu = mu / 2;
        elseif count == inner_iter
            mu = 0.5 * delta;            
        else
            break;
        end
        lambda = xi / (eigs(D-R,1));
        if(disp)
            fprintf('mu = %f, lambda = %f \n',mu, lambda)
        end
        count = 0;
    end
    
end

% Compute final objective
[obj(iter+1)] = computeObj(X,U,lpq,i,j,lambda,mu,weights,disp,iter,starttime);

% compute assignment using connected components
[clustAssign,numcomponents] = assignment(U,i,j,epsilon,nsamples);

optTime = toc(starttime);
end

%% Functions

function lpq = GemanMcClure(data, mu)
lpq = (mu ./ (mu + sum(data.^2,2))).^2;
end

function [obj] = computeObj(X,U,lpq,i,j,lambda,mu,weights,disp,iter,t)

% computing the objective as in equation [2]
data = 0.5*sum(sum((X-U).^2));
diff = sum((U(i,:) - U(j,:)).^2,2);
smooth = lambda * 0.5*((lpq.*weights)'*diff + mu * weights'*((sqrt(lpq) - 1).^2));

% final objective
obj = data + smooth;

if(disp)
    fprintf(' %3d | %f | %f | %f | %3.2f \n',iter, data, smooth, obj, toc(t))
end
end
