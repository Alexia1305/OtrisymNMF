function [G, labels] = readGraphNet(fichier,labelfind)
    % Ouvrir le fichier en lecture
    fid = fopen(fichier, 'r');
    
    if fid == -1
        error('Impossible d''ouvrir le fichier %s.', fichier);
    end
    
    % Initialisation des variables
    vertices = [];
    labels = {};  % Cellule pour stocker les labels des sommets
    edges = [];
    section = '';
    
    % Lecture du fichier ligne par ligne
    while ~feof(fid)
        ligne = strtrim(fgetl(fid));
        
        % Ignorer les lignes vides ou les commentaires
        if isempty(ligne) || startsWith(ligne, '*')
            if startsWith(ligne, '*Vertices')||startsWith(ligne, '*vertices')
                section = 'vertices';
            elseif startsWith(ligne, '*Edges')||startsWith(ligne, '*edges')
                section = 'edges';
            elseif startsWith(ligne, '*Arcs')||startsWith(ligne, '*arcs')
                section = 'arcs'; % Si c'est un graphe orienté
            end
            continue;
        end
        
        switch section
            case 'vertices'
                % Lecture des sommets : numéro du sommet et label
                tokens = regexp(ligne, '\s+', 'split', 'once');  % On sépare en deux parties (id et label)
                index = str2double(tokens{1});
                
                % Le label est entre guillemets, donc on le nettoie
                 label = strtrim(tokens{2});
                label = strrep(label, '"', '');  % Retirer les guillemets autour du label
                
                % Stocker l'index et le label
                vertices(index) = index;
                labels{index} = label;
                
            case {'edges', 'arcs'}
                % Lecture des arêtes ou arcs : sommet source, sommet cible et poids optionnel
                tokens = strsplit(ligne);
                source = str2double(tokens{1});
                cible = str2double(tokens{2});
                
                if length(tokens) == 3
                    poids = str2double(tokens{3});
                else
                    poids = 1;  % Poids par défaut
                end
                
                edges = [edges; source, cible, poids];
        end
    end
    
    fclose(fid);
    
    % Création du graphe dans MATLAB
    % Si section == 'arcs' -> graphe orienté, sinon graphe non orienté
    if strcmp(section, 'arcs')
        G = digraph(edges(:, 1), edges(:, 2), edges(:, 3));  % Graphe orienté
    else
        G = graph(edges(:, 1), edges(:, 2), edges(:, 3));     % Graphe non orienté
    end
     if labelfind==1
        G.Nodes.Name = labels(:);  % Ajouter les labels comme propriété de chaque nœud
     end 
end
