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
                section = 'arcs'; % Si c'est un graphe orient�
            end
            continue;
        end
        
        switch section
            case 'vertices'
                % Lecture des sommets : num�ro du sommet et label
                tokens = regexp(ligne, '\s+', 'split', 'once');  % On s�pare en deux parties (id et label)
                index = str2double(tokens{1});
                
                % Le label est entre guillemets, donc on le nettoie
                 label = strtrim(tokens{2});
                label = strrep(label, '"', '');  % Retirer les guillemets autour du label
                
                % Stocker l'index et le label
                vertices(index) = index;
                labels{index} = label;
                
            case {'edges', 'arcs'}
                % Lecture des ar�tes ou arcs : sommet source, sommet cible et poids optionnel
                tokens = strsplit(ligne);
                source = str2double(tokens{1});
                cible = str2double(tokens{2});
                
                if length(tokens) == 3
                    poids = str2double(tokens{3});
                else
                    poids = 1;  % Poids par d�faut
                end
                
                edges = [edges; source, cible, poids];
        end
    end
    
    fclose(fid);
    
    % Cr�ation du graphe dans MATLAB
    % Si section == 'arcs' -> graphe orient�, sinon graphe non orient�
    if strcmp(section, 'arcs')
        G = digraph(edges(:, 1), edges(:, 2), edges(:, 3));  % Graphe orient�
    else
        G = graph(edges(:, 1), edges(:, 2), edges(:, 3));     % Graphe non orient�
    end
     if labelfind==1
        G.Nodes.Name = labels(:);  % Ajouter les labels comme propri�t� de chaque n�ud
     end 
end
