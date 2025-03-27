
clear all; clc; close all;
%% Karate club network example 

% Network load
load("data/karate.mat")

%OtrisymNMF
r=2;
[w,v2,S,erreur] = OtrisymNMF_CD(A,r,'numTrials',10);
disp("NMI of OtrisymNMF partition on karate club : ")

disp(computeNMI(Label_karate,v2))

% Display network
G = graph(A);
node_degrees = degree(G);
node_sizes = 5 + 1.2 * node_degrees; 

community_colors = [1 0 0; 0 1 0; 0 0 1];  


node_colors = community_colors(v2, :);

figure;
h = plot(G); 
title('Partition by OtrisymNMF of the karate club ')
% Ajuster la taille des n�uds en fonction des degr�s
h.MarkerSize = node_sizes;
h.NodeColor = node_colors;


%% Dolphins network example


% Network
file='data/dolphins.net';
[G, labels] = readGraphNet(file,1);
clusters=ones(length(labels),1);
group2=[61,33,57,23,6,10,7,32,14,18,26,49,58,42,55,28,27,2,20,8];
for i=1:length(group2)
    clusters(group2(i))=2;
end 
num_clusters = max(clusters);

% Network plot
% G�n�rer une palette de couleurs : une couleur unique pour chaque cluster
colors = lines(num_clusters);  % 'lines' est une palette MATLAB, tu peux aussi utiliser jet, parula, etc.

% Assigner une couleur � chaque n�ud en fonction de son cluster
nodeColors = colors(clusters, :);  % Chaque ligne de nodeColors est la couleur du n�ud correspondant
figure;
% Afficher le graphe avec les couleurs assign�es
p = plot(G, 'NodeLabel',labels(:));  % Cr�er le plot du graphe
p.NodeCData = clusters;  % Utiliser les clusters comme donn�es pour les couleurs
colormap(colors);        % Appliquer la palette de couleurs
                % Afficher la barre de couleurs pour avoir une id�e des clusters
title('Dolphins Network with real partition');
X=adjacency(G);
r=2;
% OtrisymNMF

[w,v,S,erreur] = OtrisymNMF_CD(X,r);
disp("NMI of OtrisymNMF partition on Dolphins : ")
disp(computeNMI(clusters,v))

num_v = max(v);
% G�n�rer une palette de couleurs : une couleur unique pour chaque cluster
colors = lines(num_v);  % 'lines' est une palette MATLAB, tu peux aussi utiliser jet, parula, etc.
% Assigner une couleur � chaque n�ud en fonction de son cluster
nodeColors = colors(v, :);  % Chaque ligne de nodeColors est la couleur du n�ud correspondant
figure;
% Afficher le graphe avec les couleurs assign�es
p = plot(G, 'NodeLabel', labels(:));  % Cr�er le plot du graphe
title('Dolphins Network with partition find by OtrisymNMF');
p.NodeCData = v;  % Utiliser les clusters comme donn�es pour les couleurs
colormap(colors);        % Appliquer la palette de couleurs
              % Afficher la barre de couleurs pour avoir une id�e des clusters
mod=v;
vsp=v;




