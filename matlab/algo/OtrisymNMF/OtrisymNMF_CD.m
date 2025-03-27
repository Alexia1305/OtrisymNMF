% Orthogonal Symmetric NonNegative Matrix Trifactorization with a
% Coordinate descent approach
%
% function [w,v,S,error] = OtrisymNMF_CD(X,r,varargin)
%
% Heuristic to solve the following problem:
% Given a symmetric matrix X>=0, find a matrix W>=0 and a matrix S>=0 such that X~=WSW' with W'W=I,
%
% INPUTS
%
% X: symmetric nonnegative matrix nxn
% r: number of columns of W
%
% Options
% - numTrials 1*(default*) number of trials with different initializations
% - maxiter 1000* number of max iterations for a trial (number of update of W and S)
% - delta 1e-7* tolerance of the convergence error to stop the iteration
% (break if the error increase between two iterate is < delta or if the error < delta  )
% - time_limit 60*5' time limit in seconds for the heuristic 
% - init method for the initialization of the heuristic
% ("random","SSPA","SVCA","SPA") Default *SSPA for the first trial, then *SVCA
% - verbosity 1* to display messages, 0 no display
%
% OUTPUTS
%
% v: vector of lenght n, v(i) gives the index of the columns of W not nul
% for the i-th row
% w : vector of lenght n,w(i) gives the value of the non zero element of
% the i-th row
% S: central matrix rxr 
% error: relative error ||X-WSW||_F/||X||_F
% This code is a supplementary material to the paper
%Reference:  
%"Orthogonal Symmetric Nonnegative Matrix Tri-Factorization."
%2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2024.

function [w_best,v_best,S_best,erreur_best] = OtrisymNMF_CD(X,r,varargin)

if nargin <= 2
    options = [];
else
    for k = 1:2:length(varargin)
        options.(varargin{k}) = varargin{k+1};
    end
end
% Default Value 
if ~isfield(options,'numTrials')
    options.numTrials = 1;
end
if ~isfield(options,'time_limit')
    options.time_limit=5*60;
end 
if ~isfield(options,'delta')
    options.delta=5e-10;
end 
if ~isfield(options,'maxiter')
    options.maxiter=1000;
end 
if ~isfield(options,'verbosity')
    options.verbosity=1;
end 
 
tic;
erreur_best='inf';

 if options.verbosity > 0
        fprintf('Running %u Trials in Series \n', options.numTrials);
 end
for trials =1:options.numTrials
    
    %INITIALISATION
    n=size(X,1);
    w=zeros(n,1);
    v=zeros(n,1);
    if isfield(options,'init')
        init_algo=options.init;
    elseif trials==1
        init_algo="SSPA";
    elseif trials~=1
        init_algo="SVCA";
    end 
    if init_algo=="random"

        for i=1:n
            v(i)=randi([1,r]);
            w(i)=rand;
        end 
        %Normalisation des colonnes de w
        nw = zeros(1,r);
        for i = 1:n
            nw(v(i)) = nw(v(i)) + w(i)^2;
        end
        for k = 1:r
            nw(k) = sqrt(nw(k));
        end
        for i = 1:n
            w(i) = w(i)/nw(v(i));
        end
        random_matrix=rand(r,r);
        S=0.5*(random_matrix+random_matrix');
    elseif init_algo=="SSPA"
        p=max(2,floor(0.5*n/r));
        options1.average=1;
        [WO,~] = SSPA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
        for k = 1:r
            nw = norm(W(:, k), 2);
            if nw == 0
                continue;
            end
            W(:, k) = W(:, k) ./ nw;
        end
        for i=1:n 
            % Trouver l'indice du premier élément non nul dans la ligne i
            idx = find(W(i, :) ~= 0, 1);  % '1' pour récupérer le premier élément non nul

            if ~isempty(idx)
                v(i) = idx;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, idx); % Stocker la valeur de l'élément non nul
            else
                v(i) = 1;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, 1);
            end
        end 
         S=UpdateS(X,r,w,v);
    elseif init_algo=="SVCA"
        p=max(2,floor(0.1*n/r));
        options1.average=1;
        [WO,~] = SVCA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
        for k = 1:r
            nw = norm(W(:, k), 2);
            if nw == 0
                continue;
            end
            W(:, k) = W(:, k) ./ nw;
        end
        for i=1:n 
            % Trouver l'indice du premier élément non nul dans la ligne i
            idx = find(W(i, :) ~= 0, 1);  % '1' pour récupérer le premier élément non nul

            if ~isempty(idx)
                v(i) = idx;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, idx); % Stocker la valeur de l'élément non nul
            else
                v(i) = 1;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, 1);
            end
        end 
         S=UpdateS(X,r,w,v);
    elseif init_algo=="SPA"
        p=1;
        options1.average=1;
        [WO,~] = SSPA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
        for k = 1:r
            nw = norm(W(:, k), 2);
            if nw == 0
                continue;
            end
            W(:, k) = W(:, k) ./ nw;
        end
        for i=1:n 
            % Trouver l'indice du premier élément non nul dans la ligne i
            idx = find(W(i, :) ~= 0, 1);  % '1' pour récupérer le premier élément non nul

            if ~isempty(idx)
                v(i) = idx;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, idx); % Stocker la valeur de l'élément non nul
            else
                v(i) = 1;       % Stocker l'indice de l'élément non nul
                w(i) = W(i, 1);
            end
        end 
         S=UpdateS(X,r,w,v);

    end


    %UPDATE 
    erreur_prec=ComputeError(X,S,w,v);
    erreur=erreur_prec;
    for itt= 1:options.maxiter
        if toc > options.time_limit
            disp('Time limit passed');
            break;
        end
        [w,v] = UpdateW(X,S,w,v);
        erreur= ComputeError(X,S,w,v);
        S = UpdateS(X,r,w,v);
        erreur_prec=erreur;
        erreur= ComputeError(X,S,w,v);
        if erreur<options.delta
            break;
        end
        if abs(erreur_prec-erreur)<options.delta
            break;
        end

    end 
    if erreur<=erreur_best
        w_best=w;
        v_best=v;
        S_best=S;
        erreur_best=erreur;
        if erreur_best<=options.delta
            break;
        end 
         if toc > options.time_limit
            if options.verbosity>0
                fprintf('Time_limit reached \n')
            end 
            break;
        end
    end 
    if options.verbosity > 0
        if itt==options.maxiter
                fprintf('Not converged \n')
        end 
        fprintf('Trial %u of %u with %s : %2.4e | Best: %2.4e \n',...
            trials,options.numTrials,init_algo,erreur,erreur_best);
            
    end
    
end 

