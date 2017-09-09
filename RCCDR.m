function [clustAssign,numcomponents,optTime,gtlabels,nCluster] = RCCDR(filename, maxiter, inner_iter)
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
[nsamples,~] = size(X);
nCluster = length(unique(gtlabels));
npairs = size(w,1);

nfeatures = 100; % Dimension reduced feature length (User-defined)
zeta = 8;
gamma = 0.2; % regularization parameter for l1 norm
eta = 0.9; % for dictionary update

% PCA projection for initializing Z
rng(50);
[matD,~,~] = pca(X,'NumComponents',nfeatures);
Z = X*matD;

% setting weights as given in equation [S1]
R = sparse([i;j], [j;i], [ones(length(i),1);ones(length(i),1)], nsamples, nsamples);
nconn = full(sum(R,2));
weights = mean(nconn) ./ sqrt(nconn(i).*nconn(j));

% initializing U to Z and lpq = 1 \forall p,q \in E
U = Z;
lpq = ones(length(i),1);
lpq_data = ones(nsamples,1);

% computation of \delta, \delta_data, \mu
epsilon = sqrt(sum((Z(i,:) - Z(j,:)).^2,2));
% for epsilon computation:- not considering the edges with both the nodes 
% being near duplicate of each other.
epsilon(epsilon/sqrt(nfeatures) < 1e-2) = [];
[epsilon,~] = sort(epsilon);

mu = 3 * epsilon(end)^2; % setting \mu to be 3r^2

robsamp = min(250, ceil(npairs*0.01)); % top 1% of the closest neighbours
delta = mean(epsilon(1:robsamp));
epsilon = mean(epsilon(1:ceil(npairs*0.01)));

delta_data = sum(bsxfun(@minus,Z,mean(Z,1)).^2,2);
delta_data = 2 * mean(delta_data);

mu_data = max(2000, zeta*delta_data); % we set 2000 to be lower threshold

% computation of matrix A = D-R (here D is the diagonal matrix and R is
% symmetric matrix)
R = sparse([i;j], [j;i], [lpq.*weights;lpq.*weights], nsamples, nsamples);
D = sparse(1:nsamples, 1:nsamples, sum(R,2), nsamples, nsamples);
H = sparse(1:nsamples, 1:nsamples, lpq_data, nsamples, nsamples);

% initial computation of \lambda
lambda =  norm(H*Z,2) / (eigs(H,1) + eigs(D-R,1));

nu = 1;

beta = trace(Z'*Z)*0.0001; % dictionary regularization parameter

XA = Z;
DtD = matD'*matD;
spDD = eigs(DtD,1);
tau = 1 / (spDD + nu * eigs(H,1));
Zold = Z;

% structure params for eigs function
opts = struct;
opts.tol = 1e-3;
opts.maxit = 300;

count = 0;
if(disp)    
    fprintf('mu = %f, mu_data = %f, lambda = %f, epsilon = %f, delta = %f, delta_data = %f \n',mu, mu_data, lambda, epsilon, delta, delta_data)
    fprintf(' Iter | Data \t | Smooth \t | Obj \t | Time(s) \n')
end
obj = zeros(maxiter+1,1);

%% start of optimization phase
starttime = tic;

for iter = 2:maxiter+1
    
    % update lpq, lpq_data
    lpq = GemanMcClure(U(i,:)-U(j,:),mu);
    lpq_data = GemanMcClure(Z-U,mu_data);
    H = sparse(1:nsamples, 1:nsamples, lpq_data, nsamples, nsamples);
    
    % update Z using proximal gradient
    omega = (iter-2) / (iter + 1);
    Znew = Z + omega * (Z - Zold);
    Zold = Z;
    Z = Znew - tau * (-XA + Znew*DtD + nu * H * (Znew-U));
    Z = prox_l1norm(Z, tau * gamma);
    
    % compute objective
    [obj(iter)] = computeObj(Z,U,lpq,lpq_data,i,j,lambda,mu,mu_data,X,matD,beta,nu,gamma,weights,disp,iter-1,starttime);
    
    % update U's
    R = sparse([i;j], [j;i], [lpq.*weights;lpq.*weights], nsamples, nsamples);
    D = sparse(1:nsamples, 1:nsamples, sum(R,2), nsamples, nsamples);
    M = H + lambda * (D-R);
    Znew = H*Z;
    pfun = cmg_sdd(M);
    % For large number of features, computation can be easily set in parallel
    parfor k = 1:nfeatures
        [U(:,k),~] = pcg(M,Znew(:,k),1e-2,100,pfun);
    end
    
    % every 10 iteration update dictionary 
    if(mod(iter,10) == 0)
        ZZ = Z'*Z;
        beta = trace(ZZ)*0.0001;
        matAnew = ((ZZ + beta*speye(size(ZZ)))^-1 * Z'*X)';
        matD = eta * matD + (1-eta) * matAnew;
        
        XA = X*matD;
        DtD = matD'*matD;
        spDD = eigs(DtD,1);
        tau = 1 / (spDD + nu * eigs(H,1,'lm',opts));
    end
    
    % check for stopping criteria
    count = count + 1;
    if (abs(obj(iter-1)-obj(iter)) < 1e-2) || count == inner_iter
        
        if mu >= delta
            mu = mu / 2;
        elseif count == inner_iter
            mu = 0.5 * delta;
        else
            break;
        end
        if mu_data >= delta_data
            mu_data = mu_data / 2;
        else
            mu_data = 0.5 * delta_data;
        end
        count = 0;
        lambda = norm(Znew,2) / (eigs(H,1,'lm',opts) + eigs(D-R,1));
        if(disp)
            fprintf('mu = %f, mu_data = %f, lambda = %f \n',mu, mu_data, lambda)
        end
    end
    
end

% Compute final objective
[obj(iter+1)] = computeObj(Z,U,lpq,lpq_data,i,j,lambda,mu,mu_data,X,matD,beta,nu,gamma,weights,disp,iter,starttime);

% compute assignment using connected components
[clustAssign,numcomponents] = assignment(U,i,j,epsilon,nsamples);

optTime = toc(starttime);
end

%% Functions

function X = prox_l1norm(X,lambda)
X = sign(X).*max(0,abs(X) - lambda);
end

function lpq = GemanMcClure(data, mu)
lpq = (mu ./ (mu + sum(data.^2,2))).^2;
end

function [obj] = computeObj(Z,U,lpq,lpq_data,i,j,lambda,mu,mu_data,X,D,beta,nu,gamma,weights,disp,iter,t)
data2 = 0.5 * sum(sum((X - Z*D').^2));
data3 = gamma * sum(abs(Z(:))) + beta * 0.5 * norm(D,'fro')^2;

diff = sum((U(i,:) - U(j,:)).^2,2);
data1 = nu * 0.5*(lpq_data'*sum((Z-U).^2,2) + mu_data * sum((sqrt(lpq_data) - 1).^2));
smooth = nu * lambda * 0.5 * ((lpq.*weights)'*diff + mu * weights' * ((sqrt(lpq) - 1).^2));

% final objective as in equation [11]
data = data1 + data2 + data3;
obj = data + smooth;

if(disp)
    fprintf(' %3d | %f | %f | %f | %3.2f \n',iter, data, smooth, obj, toc(t))
end
end
