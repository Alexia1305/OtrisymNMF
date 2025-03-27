function [mean_score, score]= mrsa( A,B )
% this function computes the Mean Removed Spectral Angle between two
% matrices : 
% MRSA(A,B)=1/r*sum_i=1^r 100/pi*arcos(<  A(:,i)-repmat(mean(A(:,i)),m,1) ,
% B(:,i)-repmat(mean(B(:,i)),m,1)
% >/(||A(:,i)-repmat(mean(A(:,i)),m,1)||_2*||B(:,i)-repmat(mean(B(:,i)),m,1)||_2))
%
% ****** Input ****** 
% A           : m-by-r matrix (corresponding, for example, to the true basis
%               vectors)
% B           : m-by-r matrix (corresponding, for example, to the estimated
%               basis vectors)
% 
% ****** Output ****** 
% mean_score  : the gradient of (1) w.r.t. W1
% score       : the r-size vector given the MRSA column by column
%

na = size(A,2);
nb = size(B,2);
if na ~= nb
   error('Input matrices have unmatched number of columns'); 
end

for i = 1 : na
 a = A(:,i);
 b = B(:,i);
 a = a-mean(a);
 b = b-mean(b);
 
 if (norm(a,2)>0) && (norm(b,2)>0)
     a = a/norm(a,2);
     b = b/norm(b,2);

     score(i) = abs(acos(a'*b))*100/pi;
 else
     score(i)=0;
 end
  
end

 mean_score = mean(score);
end