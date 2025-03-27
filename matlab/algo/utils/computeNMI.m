function nmi = computeNMI(P_true, P_found)
    % Calcul de la NMI entre deux partitions P_true et P_found
    
    % Nombre de nœuds
    N = length(P_true);
    
    % Identifier les clusters uniques dans chaque partition
    clusters_true = unique(P_true);
    clusters_found = unique(P_found);
    
    % Calcul de la matrice de confusion
    contingency_matrix = zeros(length(clusters_true), length(clusters_found));
    
    for i = 1:length(clusters_true)
        for j = 1:length(clusters_found)
            % Nombre d'éléments dans l'intersection des clusters i et j
            contingency_matrix(i, j) = sum(P_true == clusters_true(i) & P_found == clusters_found(j));
        end
    end
    
    % Totaux pour les clusters
    sum_true = sum(contingency_matrix, 2);  % Totaux des lignes
    sum_found = sum(contingency_matrix, 1);  % Totaux des colonnes
    
    % Calcul de l'information mutuelle (MI)
    MI = 0;
    for i = 1:length(clusters_true)
        for j = 1:length(clusters_found)
            if contingency_matrix(i, j) > 0
                MI = MI + (contingency_matrix(i, j) ) * ...
                     log((N * contingency_matrix(i, j)) / (sum_true(i) * sum_found(j)));
            end
        end
    end
    
    % Calcul de l'entropie des partitions
    H_true = -sum((sum_true) .* log(sum_true / N));
    H_found = -sum((sum_found) .* log(sum_found / N));
    
    % Normalisation de l'information mutuelle
    nmi = 2 * MI / (H_true + H_found);
    if length(clusters_true)==1 && length(clusters_found)==1
        nmi=1;
    end 
end