% Smoothed Vertex Component Analysis
%
% function [W,K] = SVCA(X,r,p,options)
%
% Heuristic to solve the following problem:
% Given a matrix X, find a matrix W such that X~=WH for some H>=0,
% under the assumption that each column of W has p columns of X close
% to it (called the p proximal latent points).
%
% INPUTS
%
% X: data set of size m*n
% r: number of columns of W
% p: number of proximal latent points
%
% Options
% .lra = 1 uses a low-rank approximation of the input matrix in the
%          selection step
%      = 0 (default) does not
% .average = 1 uses the mean as an aggregation technique
%          = 0 (default) uses the median
%
% OUTPUTS
%
% W: the matrix such that X~=WH
% K: indices of the selected data points (one column per iteration)
%
% This code is a supplementary material to the paper
% Smoothed Separable Nonnegative Matrix Factorization
% by N. Nadisic, N. Gillis, and C. Kervazo
% https://arxiv.org/abs/2110.05528


function [W,K] = SVCA(X,r,p,options)

if nargin <= 3
    options = [];
end
% Low-rank approximation (LRA) of the input matrix
% Default: no low-rank approximations
if ~isfield(options,'lra')
    options.lra = 0;
end
[Y,S,Z] = svds(X,r); % Y contains the first r singular vectors of X
                     % Faster algorithms could be used; cf. ALLS
if options.lra == 1
    X = S*Z'; % Replace X by its low-rank approximation
end
% Use of the average or the median [default] to aggregate the extracted
% subsets of columns of X
if ~isfield(options,'average')
    options.average = 0;
end
% Projector (I - VV^T)  onto the orthogonal complements of the columns of W
% extracted so far.
V = [];
% Iterations of SVCA
for k = 1 : r
    % Random direction in col(Y), like in ALLS
    diru = Y*randn(r,1);
    % Projection of the random projection, for diru to be orthogonal to the
    % previously extracted columns of W
    if k >= 2
        diru = diru - V*(V'*diru);
    end
    % Inner product with the data matrix
    u = diru'*X;
    % Sorting the entries en selecting the direction maximizing |u|
    [a,b] = sort(u);
    %if abs(u(b(1))) < abs(u(b(end)))
    if abs(median(u(b(1:p)))) < abs(median(u(b(end-p+1:end))))
       b = b(end:-1:1);
    end
    % Select the indices correspondind to the largest entries of u
    K(k,:) = b(1:p);
    % Compute vertex
    if p == 1
        W(:,k) = X(:,K(k,:));
    else
        if options.average == 1
            W(:,k) = mean( X(:,K(k,:))' )';
        else
            W(:,k) = median( X(:,K(k,:))' )';
        end
    end
    % Update the projector
    V = updateorthbasis(V,W(:,k));
end
if options.lra == 1
    W = Y*W;
end