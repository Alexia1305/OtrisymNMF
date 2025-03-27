% Smoothed Successive Projection Algorithm
%
% function [W,K] = SSPA(X,r,p,options)
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


function [W,K] = SSPA(X,r,p,options);

if nargin <= 3
    options = [];
end
% Low-rank approximation (LRA) of the input matrix
% Default: no low-rank approximations
if ~isfield(options,'lra')
    options.lra = 0;
end
if options.lra == 1
    [Y,S,Z] = svds(X,r); % Other LRAs could be used
    Z = S*Z';
else
    Z = X;
end
% Use of the average or the median [default] to aggregate the extracted
% subsets of columns of X
if ~isfield(options,'average')
    options.average = 0;
end
% Projector onto the orthogonal complements of the columns of X extracted
% so far, or Z if using an LRA of X as a preprocessing,
% P = (I - VV^T)
V = [];
% To get the SPA direction, we will use the recursive formula:
% for any u s.t. ||u||_2 = 1: ||(I-uu^T) v||_2^2 = ||v||_2^2 - u' * v.
normX2 = sum(X.^2);
% Iterations of SSPA
for k = 1 : r
    % Select SPA direction
    [spa,spb] = max( normX2 );
    diru = X(:,spb) ;
    % Projection of the SPA projection, for diru to be orthogonal to the
    % previously extracted columns of W
    if k >= 2
        diru = diru - V*(V'*diru);
    end
    % Inner product with the data matrix
    u = diru'*X;
    % Sorting the entries en selecting the direction maximizing u
    [a,b] = sort(u,'descend');
    % Select the indices correspondind to the largest entries of u
    K(k,:) = b(1:p);
    % Compute vertex
    if p == 1
        W(:,k) = Z(:,K(k,:));
    else
%         if options.average == 1
%             W(:,k) = mean( Z(:,K(k,:))' )';
%         else
%             W(:,k) = median( Z(:,K(k,:))' )';
%         end
        if options.average == 1
            W(:,k) = mean( Z(:,K(k,:))' )';
        elseif options.average == 3
            W(:,k) = colaverage( Z(:,K(k,:)) , 3 )';
        elseif options.average == 4
            W(:,k) = colaverage( Z(:,K(k,:)) , 4 )';
        else
            W(:,k) = median( Z(:,K(k,:))' )';
        end
    end
    % Update the projector
    V = updateorthbasis(V,W(:,k));
    % Update the squared l2 norm of the columns of (I-VV^T)X
    normX2 = normX2 - (V(:,end)'*X).^2;
end
if options.lra == 1
    W = Y*W;
end