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
% X: symmetric nonnegative matrix nxn sparse 
% r: number of columns of W
%
% Options
% - numTrials 1*(default*) number of trials with different initializations
% - init method for the initialization of the heuristic
% ("random","SSPA","SVCA","SPA") Default *SSPA for the first trial, then *SVCA
% - update_rule "original"* or "S_direct"
% - maxiter 1000* number of max iterations for a trial (number of update of W and S)
% - delta 1e-7* tolerance of the convergence error to stop the iteration
% (break if the error increase between two iterate is < delta or if the error < delta  )
% - time_limit 60*5' time limit in seconds for the heuristic 
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
%  TOCOMPLETE 

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
if ~isfield(options,'update_rule')
    options.update_rule="original";
end
 
tic;
erreur_best='inf';

 if options.verbosity > 0
        fprintf('Running %u Trials in Series \n', options.numTrials);
 end
 
% PRECOMPUTATION 
n         = size(X,1);
normX     = norm(X,'fro');
normX2    = normX^2;
[I,J,V]   = find(X);    
[~, perm] = sort(I,'ascend'); % permutation to sort given W rows
I         = I(perm);
J         = J(perm);
V         = V(perm);
rowStart  = [1; find(diff(I))+1; numel(I)+1];
 
 
for trials =1:options.numTrials
    
    %INITIALISATION
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
        
    elseif init_algo=="SSPA"
        p=max(2,floor(0.1*n/r));
        options1.average=1;
        [WO,~] = SSPA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
          
        % construction of v and w given W
        v=max((W~=0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
        
    elseif init_algo=="SVCA"
        p=max(2,floor(0.1*n/r));
        options1.average=1;
        [WO,~] = SVCA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
        
        % construction of v and w given W
        v=max((W~=0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
        
%         for i=1:n 
%             % Trouver l'indice du premier élément non nul dans la ligne i
%             idx = find(W(i, :) ~= 0, 1);  % '1' pour récupérer le premier élément non nul
% 
%             if ~isempty(idx)
%                 v(i) = idx;       % Stocker l'indice de l'élément non nul
%                 w(i) = W(i, idx); % Stocker la valeur de l'élément non nul
%             else
%                 v(i) = 1;       % Stocker l'indice de l'élément non nul
%                 w(i) = W(i, 1);
%             end
%         end 
        
    elseif init_algo=="SPA"
        p=1;
        options1.average=1;
        [WO,~] = SSPA(X,r,p,options1);
        norm2x = sqrt(sum(X.^2, 1));
        Xn = X .* (1 ./ (norm2x + 1e-16));

        HO = orthNNLS(X, WO, Xn);
        W = HO';
        
          
        % construction of v and w given W
        v=max((W~=0).*(1:r),[],2);
        zero_idx = find(v == 0);
        v(zero_idx) = randi(r, size(zero_idx)); 
        w = W(sub2ind(size(W), (1:n)', v));
         

    end
    
    % Normalization of w 
     colNorm = sqrt(accumarray(v,w.^2,[r 1]));
    w       = w./(colNorm(v)+1e-10);
    
    % Construction of S
    prodVal = w(I).*w(J).*V; %calcul des termes w_i*w_j*X(i,j)
    S       = accumarray([v(I),v(J)],prodVal,[r,r]);
    dgS     = diag(S)';
    
    %Matrix G = W^TXW  et  d = ||W(:,k)||^2
    d = ones(1,r);
    G = S;


    %UPDATE 
    erreur_prec= sqrt(normX2-norm(S,'fro')^2)/normX;
    erreur=erreur_prec;
    
    
    for itt= 1:options.maxiter
        
        if toc > options.time_limit
            disp('Time limit passed');
            break;
        end
        
        if options.update_rule=="S_direct" %%%%%%%%% UPDATE S DIRECT VERSION %%%%%
            
            % Precomputations
            p  = zeros(1,r);
            S2 = S.^2;
            for k = 1:r
                p(k) = sum(w.^2 .* S2(v,k));
            end

            % W Update : 
            for i = 1 : n
                %b coefficients of the r problems  min_x ax^4+bx^2+cx
                tempB = (w(i)/d(v(i)))*(G(v(i),:)./d); %w(i)*S(v(i),:)
                b     = 2*(p-tempB.^2)-2*X(i,i)*dgS; % O(r)

                %c coefficients of the r problems  min_x ax^4+bx^2+cx
                ind    = rowStart(i):rowStart(i+1)-1; % indices nonzero X(i,:) 
                mask   = J(ind) ~= i;                 % without
                cols_i = J(ind(mask));                % indices  nonzero  X(i,:) without i !
                xip    = V(ind(mask));                %  X values 

                tempC = ( xip(:) .* w(cols_i) ) ./ d( v(cols_i) ).'; %O(nnzX(i,:))
                Gscl  = bsxfun(@rdivide,G( v(cols_i), : ), d);
                c     = -4 * ( tempC' * Gscl );     % O( r*(nnz in X(i,:)) )


                % Solve r problems  min_x ax^4+bx^2+cx
                best_f = inf; best_x = 0; best_k = 1;
                for k = 1:r
                    S2kk = (G(k,k)/(d(k)^2))^2;
                    x = cardan_depressed(4*S2kk,0,2*b(k),c(k));
                    f = S2kk*x^4 + b(k)*x^2 + c(k)*x;
                    if f < best_f
                        best_f = f; best_x = x; best_k = k;
                    end
                end

                % Precomputation
                G_old     = G(v(i),:);             
                G_best    = G(best_k,:);            
                oldRow_p  = (G_old.^2)  ./ ( d(v(i)) * (d.^2) );
                bestRow_p = (G_best.^2) ./ ( d(best_k)* (d.^2) );


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Mise à jour de d (les normes au carré)
                d(v(i))   = d(v(i))  - w(i)^2;
                d(best_k) = d(best_k) + best_x^2;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % G Update
                coeffVec   = xip(:).*w(cols_i);

                % Old contribution
                delta_old       = accumarray(v(cols_i), w(i) * coeffVec, [r 1]).';
                delta_old(v(i)) = 2*delta_old(v(i));
                G(v(i),:)       = G(v(i),:) - delta_old;
                G(:,v(i))       = G(v(i),:).';

                % New contribution
                delta_new         = accumarray(v(cols_i), best_x * coeffVec, [r 1]).';
                delta_new(best_k) = 2*delta_new(best_k);
                G(best_k,:)       = G(best_k,:) + delta_new;
                G(:,best_k)       = G(best_k,:).';

                % Diagnoal values 
                G(v(i),v(i))      = G(v(i),v(i)) - w(i)^2 * X(i,i);
                G(best_k,best_k)  = G(best_k,best_k) + best_x^2 * X(i,i);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Update p
                %  p(k) = sum_j w_j^2*S(v(j),k)^2 = (sum_c G(c,k)^2/d(c) )/d(k)^2
                newRow_old   = ( G(v(i) , : ).^2 ) ./ ( d(v(i))  * (d.^2) );
                newRow_best  = ( G(best_k, : ).^2 ) ./ ( d(best_k) * (d.^2) );

                if v(i) ~= best_k
                    p = p - oldRow_p  - bestRow_p + newRow_old + newRow_best;
                else
                    p = p - oldRow_p + newRow_old;
                end

                tmp = G(:,v(i)).^2 ./ d.';     
                p(v(i))  = sum(tmp) / (d(v(i))^2);

                tmp = G(:,best_k).^2 ./ d.';
                p(best_k) = sum(tmp) / (d(best_k)^2);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %update dgS
                dgS(v(i))  = G(v(i),v(i))  / (d(v(i))^2);
                dgS(best_k) = G(best_k,best_k) / (d(best_k)^2);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%%%%%%%%%%%%%%%
                %update w(i)
                w(i) = best_x;
                v(i) = best_k;
                %%%%%%%%%%%%%%%%%
            end


            %Normalization 
            % S = W^TXW when W^TW=I
            colNorm = sqrt(accumarray(v,w.^2,[r 1]));
            w       = w./colNorm(v);

            %Construction S O(nnz(X))
            prodVal = w(I).*w(J).*V; % O(nnz(X))
            S       = accumarray([v(I),v(J)],prodVal,[r,r]); % O(nnz(X) + r^2)
            dgS     = diag(S)';

            % G = W^TXW  et  d = ||W(:,k)||^2
            d = ones(1,r);
            G = S;
            
        else  %%% ORGINAL VERSION %%%
        
            % Precomputation
            p  = zeros(1,r);
            S2 = S.^2;
            for k = 1:r
                p(k) = sum(w.^2 .* S2(v,k));
            end

            % UPDATE W
            for i = 1 : n
                %b coefficients of the r problems  min_x ax^4+bx^2+cx
                b = 2*(p-(w(i)*S(v(i),:)).^2)-2*X(i,i)*dgS; % O(r)

                %c coefficients of the r problems  min_x ax^4+bx^2+cx
                ind    = rowStart(i):rowStart(i+1)-1; % indices nonzeros X(i,:) 
                mask   = J(ind) ~= i;                 % remove i index
                cols_i = J(ind(mask));                % indices nonzeros de X(i,:) without i !
                xip    = V(ind(mask));                %  X values for non zeros entries indices X(i,:) without i !
                c      = -4 * ( (xip(:).*w(cols_i))' * S( v(cols_i) , : ) );   % O( nnz dans X(i,:) )

                % Solve r problems  min_x ax^4+bx^2+cx
                best_f = inf; best_x =sqrt(r/n) ; best_k = 1;
                for k = 1:r
                    x = cardan_depressed(4*S2(k,k),0,2*b(k),c(k));
                    if x<=0
                        x=sqrt(r/n);
                    end

                    f = S2(k,k)*x^4 + b(k)*x^2 + c(k)*x;
                    if f < best_f
                        best_f = f; best_x = x; best_k = k;
                    end
                end

                % Update of p before updating w(i) (O(r))
                p = p - (w(i)*S(v(i),:)).^2 + (best_x*S(best_k,:)).^2;

                % Update w(i)
                w(i) = best_x;
                v(i) = best_k;
            end

            %Normalization of w O(n)

            colNorm = sqrt(accumarray(v,w.^2,[r 1]));
            w       = w./(colNorm(v)+1e-10);

            %Construction of S O(nnz(X)) S = W^TXW when W^TW=I
            prodVal = w(I).*w(J).*V; % calcul w_i*w_j*X(i,j) O(nnz(X))
            S       = accumarray([v(I),v(J)],prodVal,[r,r]); % O(nnz(X) + r^2)
            dgS     = diag(S)';
      
            
        end
        
        
        erreur_prec=erreur;
        erreur=sqrt(normX2-norm(S,'fro')^2)/normX;
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

