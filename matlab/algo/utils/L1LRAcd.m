% Cyclic coordinate descent method for l1 LRA 
% 
% Given an m-by-n matrix and a factorization rank r, this code tries to solve
% 
%   min_{U m-by-r, V r-by-n}  ||M-UV||_1 = sum_ij |M-UV|_ij  (l1-LRA)
%
% upding each column of U and each row V sequentially; see 
% Robust Subspace Computation Using L1 Norm, Q. Ke and T. Kanade, 2003,
% http://www.cs.cmu.edu/afs/.cs.cmu.edu/Web/People/ke/publications/CMU-CS-03-172.pdf
%
% The subproblems can be solved exactly (minimization of a piece-wise linear 
% function, which is equivalent to a weighted median problem) that we 
% solve using a modification of the code from the following paper    
% "Dimensionality Reduction, Classification, and Spectral Mixture Analysis 
% using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
% Optical Engineering 50, 027001, February 2011.
% Available on http://sites.google.com/site/nicolasgillis/code
% 
% [x,y] = L1LRAcd(M,r,maxiter)
%
% Input.
%   M              : (m x n) matrix to factorize.
%   r              : factorization rank, default = 1.
%   maxiter        : number of iterations, default = 100.
%   U0, V0         : initial matrices, default: truncated SVD solution
%                                      (--> optimal solution for the l2-norm problem)
%
% Output.
%   (U,V) : approximate solution to min_{U, V} ||UV^T - M||_M, 
% 
% This code was implemented by N. Gillis for running numerical experiments
% for the paper: 
% N. Gillis and S. Vavasis, On the Complexity of Robust PCA and l1-norm Low-Rank 
% Matrix Approximation, 2015. 

function [U,V,e] = L1LRAcd(M,r,maxiter,U0,V0)

[m,n] = size(M);
if nargin <= 1, r = 1; end
if nargin <= 2, maxiter = 100; end
% Initialization with the truncated SVD
if nargin <= 3
    [u,s,v] = svds(M,r); 
    U = u*s; 
    V = v'; 
else
    U = U0; V = V0; 
end
% Coordinate descent: update of the columns of U and rows of V sequentially 
%fprintf('Coordinate descent method for l1-LRA started... \n')
for i = 1 : maxiter 
    R = M - U*V; % total residue
    for k = 1 : r
        % Current residue
        R = R + U(:,k)*V(k,:); 
        % Weighted median subproblems  
        U(:,k) = wmedian(R,V(k,:)');
        V(k,:) = wmedian(R',U(:,k))';
        % Update total residue
        R = R - U(:,k)*V(k,:); 
    end
    if nargout >= 3
        e(i) = sum(sum(abs(R)));
    end
%     if mod(i,100) == 0, fprintf('%1.0f...\n',i); 
%     elseif mod(i,10) == 0, fprintf('%1.0f...',i); end
end

% WMEDIAN computes an optimal solution of
%
% min_x  || A - xy^T ||_1 
%
% where A has dimension (m x n), x (m) and y (n),
% in O(mn log(n)) operations. Note that it can be done in O(mn). 
% 
% This code comes from the paper 
% "Dimensionality Reduction, Classification, and Spectral Mixture Analysis 
% using Nonnegative Underapproximation", N. Gillis and R.J. Plemmons,
% Optical Engineering 50, 027001, February 2011.
% Available on http://sites.google.com/site/nicolasgillis/code

function x = wmedian(A,y)

% Reduce the problem for nonzero entries of y
indi = abs(y) > 1e-16; 
A = A(:,indi);
y = y(indi); 
[m,n] = size(A);
A = A./repmat(y',m,1);
y = abs(y)/sum(abs(y));

% Sort rows of A, m*O(n log(n)) operations
[As,Inds] = sort(A,2);

% Construct matrix of ordered weigths
Y = y(Inds);

% Extract the median
actind = 1:m;
i = 1; 
sumY = zeros(m,1);
x = zeros(m,1);
while ~isempty(actind) % O(n) steps... * O(m) operations
    % sum the weitghs
    sumY(actind,:) = sumY(actind,:) + Y(actind,i);
    % check which weitgh >= 0
    supind = (sumY(actind,:) >= 0.5);
    % update corresponding x
    x(actind(supind)) = As(actind(supind),i);
    % only look reminding x to update
    actind = actind(~supind);
    i = i+1;
end