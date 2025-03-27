% Different ways to 'average' a subset of columns of a matrix
% 
% type = 1 : average 
% type = 2 : median 
% type = 3 : scaled singular vector, that is, w is the solution of 
%            min_{x,y} ||W-wy^T||_F^2 such that mean(y) = 1. 
% type = 4 : scaled l1 singular vector, that is, w is the solution of 
%            min_{x,y} ||W-wy^T||_1 such that median(y) = 1. 

function w = colaverage(W,type)

if size(W,2) == 1
    w = W;
else
    if type == 1
        w = mean(W')'; 
    elseif type == 2
        w = median(W')'; 
    elseif type == 3
        [u,s,v] = svds(W,1); 
        if sum(v(v>0)) < sum(v(v<0))
            u = -u; v = -v;
        end
        v = s*v; 
        w = u*mean(s*v); 
    elseif type == 4
        [u,v] = L1LRAcd(W,1,100); 
%         if sum(v(v>0)) < sum(v(v<0))
%             u = -u; v = -v;
%         end
%         v = s*v; 
%         w = u*median(s*v); 
        w = u*median(v);
    end
end