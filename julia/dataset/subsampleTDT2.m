clear all; clc; 
load tdt2_top30
X = X';
[m,n] = size(X);
r = 20;
rng(2020);
[W,H] = FroNMF(X,r);
% Subampled words (rows of X)
K = []; L = [];
maxowrdspertop = 20; % number of words kept per topic
maxodocspertop = 50; % number of docs kept per topic
for i = 1 : r
    [~,coli] = sort(W(:,i),'descend');
    K = unique([K; coli(1:maxowrdspertop)]);
    [~,rowi] = sort(H(i,:)','descend');
    L = unique([L; rowi(1:maxodocspertop)]);
end

Xkl = X(K,L); 
size(Xkl)