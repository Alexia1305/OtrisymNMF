function [w,v] = UpdateW(X,S,w,v)
    n = size(X,1);
    r = size(S,1);
    
    %Pré-calculs afin d'éviter une double boucle sur "n" par après.
    %O(nr)
    wp2 = zeros(1,r);
    for k = 1 : r
        for p = 1 : n
            wp2(k) = wp2(k) + (w(p)*S(v(p),k))^2;
        end
    end
    
    %Mise à jour de W : 
    % - pour chacune des lignes i=1,...,n,
    %   - pour chacune des entrées W(i,k) avec k=1,...,r
    %     - on calcule la valeur optimale de W(i,k) en supposant
    %       les entrées W(i,j)=0 pour tout j!=k
    %   - on garde la meilleure des r possibilités
    %Au total : O( nnz(X)r + nr ) (à vérifier minutieusement !)
    for i = 1 : n
        %On teste les r possibilités
        vi_new = -1;
        wi_new = -1;
        f_new  = Inf;
        for k = 1 : r
            %minimisation de c3x^4+c1*x^2+c0*x
            %c-à-d résolution de 4*c3*x^3+2*c1*x+c0 = 0 et identification
            %de la bonne racine parmi les trois racines éventuelles
            c3 = S(k,k)^2;
            c2 = 0;
            c1 = 2*(wp2(k)-(w(i)*S(v(i),k))^2)-2*S(k,k)*X(i,i);
            c0 = 0;
            % Obtenir les indices des colonnes non nulles dans la ligne i de X
            cols = find(X(i, :) ~= 0);

            % Parcourir uniquement les colonnes non nulles de la ligne i
            for idx = 1:length(cols)
                p = cols(idx);
                if p ~= i
                    c0 = c0 + X(i, p) * w(p) * S(v(p), k); % c0 = c0 + X(i, p) * WS(p, k);
                end
            end
%             for p = 1 : n
%                 if p~=i && X(i,p)~=0 %ici où la sparsité de X joue !
%                     c0 = c0 + X(i,p)*w(p)*S(v(p),k); %c0 = c0 + X(i,p)*WS(p,k);
%                 end
%             end
            c0 = -4*c0;
            roots = cardan(4*c3,c2,2*c1,c0);
            % Initialiser la solution et la valeur minimale
            % valeur par défaut si pas de minimum positif 
            x = sqrt(r/n);
            min_value = c3*x^4 + c1*x^2 + c0*x;
           

            % Parcourir toutes les racines
            for j = 1:length(roots)
                sol= roots(j);
                value = c3 * (sol ^ 4) + c1 * (sol ^ 2) + c0 * sol;
                if sol > 0 && value < min_value
                    x = sol;
                    min_value = value;
                end
            end
            if c3*x^4+c1*x^2+c0*x < f_new
                f_new  = c3*x^4+c1*x^2+c0*x;
                wi_new = x;
                vi_new = k;
            end
        end
        
        %Avant l'update de w(i), on met à jour wp2 ! (en O(r) donc pas grave)
        for k = 1 : r
            wp2(k) = wp2(k) - (w(i)*S(v(i),k))^2 + (wi_new*S(vi_new,k))^2;
        end
        %update de w(i)
        w(i) = wi_new;
        v(i) = vi_new;
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
end