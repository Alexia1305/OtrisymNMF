include("../algo/OtrisymNMF.jl")
include("../algo/OtrisymNMF_sparse.jl")
using DelimitedFiles
using Plots
gr()  # Configurer le backend GR
using SparseArrays
using LaTeXStrings
using Random
Random.seed!(2001)


n=200
r=8
level=0.6
        
W_true2 = zeros(n, r)
for i in 1:n
    k = rand(1:r)
    W_true2[i, k] = rand()+1
end

# Supprimer les colonnes nulles
# Trouver les indices des colonnes non-nulles
indices_colonnes_non_nulles = findall(x -> any(x .!= 0), eachcol(W_true2))

# Extraire les colonnes non-nulles
W_true = W_true2[:, indices_colonnes_non_nulles]
r = size(W_true, 2)
#normaliser 
for j in 1:r
    W_true2[:, j] .= W_true2[:, j] ./ norm(W_true2[:, j],2)
end
# Densité de la matrice (proportion d'éléments non nuls)
density = 0.3

# Générer une matrice sparse aléatoire
random_sparse_matrix = sprand(r, r, density)
S=Matrix(random_sparse_matrix)
S = 0.5 * (S + transpose(S))
# Mettre les éléments diagonaux à 1
for k in 1:r
    S[k,k]=1
end 
X = W_true * S* transpose(W_true)

#ajout du bruit 
Xbef=X
N = randn(n,n); 
Nbef=N;
println(norm(N,:fro))
N = level * (N/norm(N))*norm(X);  

X=X+N
X=max.(X, 0) # pas de vaelurs négatives 

maxiter=10000
epsi=10e-5
# algorithme :

W, S, erreur = OtrisymNMF_CD(X, r, maxiter,epsi,"sspa")
println(erreur)
println(calcul_accuracy(W_true,W))
v,w, S, erreur=OtrisymNMF_CD_sparse(sparse(X), r, maxiter,epsi,"sspa")
W=zeros(n,r)
for k in 1:n
    W[k,v[k]]=w[k]
end 
println(erreur)
println(calcul_accuracy(W_true,W))
matwrite("data.mat", Dict("X" => Xbef, "N" => Nbef,"ptue"=>v))
     
        
        