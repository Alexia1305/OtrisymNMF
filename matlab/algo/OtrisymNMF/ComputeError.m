function f = ComputeError(X,S,w,v)
    n = size(X,1);
    f = 0;
    %Xt = zeros(n,n);
    [i, j, val] = find(X); 

    % Parcourir uniquement les éléments non nuls de X
    for k = 1:length(val)
         f = f + (val(k)-S(v(i(k)),v(j(k)))*w(i(k))*w(j(k)))^2;
    end 
%     for i = 1 : n
%         for j = 1 : n
%             f = f + (X(i,j)-S(v(i),v(j))*w(i)*w(j))^2;
%             %Xt(i,j) = S(v(i),v(j))*w(i)*w(j);
%         end
%     end
    %Xt
    f = sqrt(f)/norm(X,'fro');
end