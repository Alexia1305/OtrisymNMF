function S = UpdateS(X,r,w,v)
    % Effectuer S = W'*(X*W) co�te O(n^2r)
    % Cependant, il y a moyen de faire le produit en O(n^2)
    % et m�me O(nnz(X)) car si on avait X sous forme sparse, on pourrait �viter les deux boucles sur n :
    S = zeros(r,r);
   
    % R�cup�rer les indices et les valeurs non nulles de la matrice sparse X
    [i, j, val] = find(X); 

    % Parcourir uniquement les �l�ments non nuls de X
    for k = 1:length(val)
        S(v(i(k)), v(j(k))) = S(v(i(k)), v(j(k))) + w(i(k)) * w(j(k)) * val(k);
    end
    
    S=max(0,S); % some tests with negative X
    
%     for i = 1 : n
%         for j = 1 : n
%             if X(i,j)~=0 %ici o� la sparsit� de X joue !
%                 S(v(i),v(j)) = S(v(i),v(j)) + w(i)*w(j)*X(i,j);
%             end
%         end
%     end

end