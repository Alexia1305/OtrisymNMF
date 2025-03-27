% Given a matrix V with orthogonal columns and a vector v,
% project v onto

function V = updateorthbasis(V,v);

if isempty(V)
    V = v/norm(v);
else
    % Project new vector onto orthogonal complement, and normalize
    v = v - V*(V'*v);
    v = v/norm(v);
    V = [V v];
end