moyenne=zeros(4,7);
ecart=zeros(4,7);
j=0;
g_r=[2,5,10,20,30,40,50]
meilleurs=zeros(4,7);
for r=g_r
    j=j+1;
    % Définir le nom du fichier .mat en fonction de la valeur de r
    nom_fichier = sprintf('CBCL_cluster_%d.mat', r);

    % Charger les données à partir du fichier .mat
    data = load(nom_fichier);

    % Accéder aux listes de sous-listes
    label_CD = data.label_CD;
    label4_K = data.label4_K;
    label3_ONMF = data.label3_ONMF;
    label2_MU = data.label2_MU;
    label_true=data.label_true;
    n_label_CD = length(label_CD);
    accu=zeros(4,n_label_CD);
    for i = 1:n_label_CD
       
        
        L=label_true{i};
        L2 = label_CD{i};
        Lmap = bestMap(L,L2);
        accu(1,i) =100* sum(L(:) == Lmap(:)) / length(L);
        L2 = label2_MU{i};
        Lmap = bestMap(L,L2);
        accu(2,i) = 100*sum(L(:) == Lmap(:)) / length(L);
        L2 = label3_ONMF{i};
        Lmap = bestMap(L,L2);
        accu(3,i) = 100*sum(L(:) == Lmap(:)) / length(L);
        L2 = label4_K{i};
        Lmap = bestMap(L,L2);
        accu(4,i) =100* sum(L(:) == Lmap(:)) / length(L); 
        maxi=max(accu(:,i));
        indices = find(accu(:,i) == maxi);
        meilleurs(indices,j)= meilleurs(indices,j)+1;
        % Trouver l'indice de l'élément maximum pour chaque ligne
        
        
    end
    meilleurs(:,j)= meilleurs(:,j)./n_label_CD;
    disp(r)
    moyenne(:,j)= mean(accu,2);
    ecart(:,j)= std(accu,0,2);
    
    
end 
meilleurs= meilleurs.*100;
% Définir le nom du fichier avec r dans le nom
nom_fichier = 'resultats.mat';

% Enregistrer les variables dans un fichier .mat
save(nom_fichier, 'g_r', 'moyenne', 'ecart');

% Plot pour CD
errorbar(g_r, moyenne(1,:),ecart(1,:), '-o', 'DisplayName', 'CD');
hold on;

% Plot pour MU
errorbar(g_r,moyenne(2,:),ecart(2,:), '-s', 'DisplayName', 'MU');

% Plot pour ONMF
errorbar(g_r, moyenne(3,:), ecart(3,:), '-d', 'DisplayName', 'ONMF');

% Plot pour Kmeans
errorbar(g_r, moyenne(4,:), ecart(4,:), '-^', 'DisplayName', 'Kmeans');

% Réglages de la figure
xlabel('r', 'Interpreter', 'latex');
ylabel('Accuracy', 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex');
grid on;

% Style LaTeX
set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'FontSize', 12);
set(gcf, 'Color', 'w');
% Sauvegarde de la figure
saveas(gcf, 'figure_accuracy.pdf');
% Création de la figure
figure;

% Plot pour CD
plot(g_r, moyenne(1,:), '-', 'DisplayName', 'CD', 'Color', 'b','Marker', 'o');
hold on;

% Plot pour MU
plot(g_r, moyenne(2,:), '--', 'DisplayName', 'MU', 'Color', 'r','Marker', 'o');

% Plot pour ONMF
plot(g_r, moyenne(3,:), '-.', 'DisplayName', 'ONMF', 'Color', 'g','Marker', 'o');

% Plot pour Kmeans
plot(g_r, moyenne(4,:), ':', 'DisplayName', 'Kmeans', 'Color', 'm','Marker', 'o');

% Réglages de la figure
xlabel('r');
ylabel('Accuracy');

% Limiter la plage des valeurs sur l'axe des abscisses jusqu'à 100
ylim([50, 100]);
legend('Location', 'best');
grid on;

% Style LaTeX
set(gca, 'TickLabelInterpreter', 'latex');
set(gca, 'FontSize', 12);
set(gcf, 'Color', 'w');

% Sauvegarde de la figure
saveas(gcf, 'figure_accuracy2.pdf');
