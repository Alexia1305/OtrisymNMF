using Random
using LinearAlgebra
using IterTools
using Combinatorics
using Clustering
using Hungarian 
using Random

Random.seed!(1313)


include("ONMF.jl")
include("SSPA.jl")


function fourth_degree_polynomial(a, b, c, d,e,x)
    return a*x^4+b*x^3+c*x^2+d*x+e
end

function roots_third_degree(a, b, c, d)
    
    if a == 0
        if b == 0
            root1 = -d/c
            return root1
        end
        delta = c^2-4*b*d
        root1 = (-c + sqrt(delta))/ (2 * b)
        root2 = (-c - sqrt(delta))/ (2 * b)
        if root1 == root2
            return [root1]
        else
            return [root1, root2]
        end
    end

    p = -(b^2 / (3 * a^2)) + c / a
    q = ((2 * b^3) / (27 * a^3)) - ((9 * c * b) / (27 * a^2)) + (d / a)
    delta = -(4 * p^3 + 27 * q^2)
    if delta < 0
        u = (-q + sqrt(-delta / 27)) / 2
        v = (-q - sqrt(-delta / 27)) / 2
        if u < 0
            u = -(-u)^(1 / 3)
        elseif u > 0
            u = u^(1 / 3)
        else
            u = 0
        end
        if v < 0
            v = -(-v)^(1 / 3)
        elseif v > 0
            v = v^(1 / 3)
        else
            v = 0
        end
        root1 = u + v - (b / (3 * a))
        return [root1]
    elseif delta == 0
        if p == q == 0
            root1 = 0
            return [root1]
        else
            root1 = (3 * q) / p
            root2 = (-3 * q) / (2 * p)
            return [root1,root2]
        end
    else
        epsilon = -1e-300
        phi = acos(-q / 2 * sqrt(-27 / (p^3 + epsilon)))
        z1 = 2 * sqrt(-p / 3) * cos(phi / 3)
        z2 = 2 * sqrt(-p / 3) * cos((phi + 2 * π) / 3)
        z3 = 2 * sqrt(-p / 3) * cos((phi + 4 * π) / 3)
        root1 = z1 - (b / (3 * a))
        root2 = z2 - (b / (3 * a))
        root3 = z3 - (b / (3 * a))
        return [root1, root2, root3]
    end
end

function minimize_degre4(a,b,c,d,e)
    roots=roots_third_degree(4*a,3*b,2*c,d)
    min_x=0
    min_value=e
    for root in roots
        if root >=0
            value=fourth_degree_polynomial(a, b, c, d,e,root)
            if value<=min_value
                min_x=root
                min_value=value
            end
        end
    end 
    return min_x,min_value
end 





function calcul_erreur(X, W, S,lambda) # non relative 
    n = size(W, 2)
    result = norm(X - W * S * W')^2
    for i in 1:n
        for j in 1:n
            if i != j
                result += lambda*dot(W[:, i], W[:, j])
            end
        end
    end
    return result
end

function UpdateW(X,W,S,lambda)
    
    n,r=size(W)
    SWT=S*W'
    WSWT=W*SWT
    sumW=sum(W, dims=2)
    for p in 1:r
        
        for i in 1:n
            
            # retire l'ancienne valeur dans les pré calculs
            sumW[i]-=W[i,p]
            indices_sauf_i = filter(x -> x != i, 1:n)
            WSWT[i,indices_sauf_i].-=W[i,p]*(SWT[p,indices_sauf_i])
            WSWT[:,i].=WSWT[i,:]
            SWT[:,i].-=S[:,p]*W[i,p]'
            WSWT[i,i]-=2*SWT[p,i]*W[i,p]+W[i,p]^2*S[p,p]
           
           
            C=0
            D=0
            E=0
        
            for j in 1:n
                if j !=i
                    Cj=SWT[p,j]
                    Dj=WSWT[i,j]
                    E+=Cj^2
                    C+=Cj*(Dj-X[i,j])
                    D+=(X[i,j]-Dj)^2
                end 
            end 

            A=SWT[p,i]
            B=WSWT[i,i]

            a=S[p,p]^2
            b=4*A*S[p,p]
            c=4*A^2+2*S[p,p]*(B-X[i,i])+2*E
            d=4*A*(B-X[i,i])+4*C+2*lambda*(sumW[i])
            e=(X[i,i]-B)^2+2*D

            W[i,p],value=minimize_degre4(a,b,c,d,e)



        
        

            # mise à jour des pré calculs :
            sumW[i]+=W[i,p]
            WSWT[i,i]+=2*SWT[p,i]*W[i,p]+W[i,p]^2*S[p,p]
            SWT[:,i].+=S[:,p]*W[i,p]'
            WSWT[i,indices_sauf_i].+=W[i,p]*(SWT[p,indices_sauf_i])
            WSWT[:,i].=WSWT[i,:]

           

        end 
    end
    return W
end 

function UpdateW2(X,W,S,lambda)
    
    n,r=size(W)
    SWT=S*W'
    WSWT=W*SWT
    sumW=sum(W, dims=2)
    for i in 1:n
        value_min=+ Inf
        x_min=1
        ind_min=1
        
        for p in 1:r
            
            
            C=0
            D=0
            E=0
        
            for j in 1:n
                if j !=i
                    Cj=SWT[p,j]
                    Dj=WSWT[i,j]-W[i,p]*(SWT[p,j])
                    E+=Cj^2
                    C+=Cj*(Dj-X[i,j])
                    D+=(X[i,j]-Dj)^2
                end 
            end 

            A=SWT[p,i]-S[p,p]*W[i,p]'
            B=WSWT[i,i]-2*A*W[i,p]-W[i,p]^2*S[p,p]

            a=S[p,p]^2
            b=4*A*S[p,p]
            c=4*A^2+2*S[p,p]*(B-X[i,i])+2*E
            d=4*A*(B-X[i,i])+4*C+2*lambda*(sumW[i]-W[i,p])
            e=(X[i,i]-B)^2+2*D

            x,value=minimize_degre4(a,b,c,d,e)
            if value <= value_min
                value_min=value
                ind_min=p
                x_min=x
            end 


        
        

            

           

        end 
        
        # retire l'ancienne valeur dans les pré calculs
        sumW[i]-=W[i,ind_min]
        indices_sauf_i = filter(x -> x != i, 1:n)
        WSWT[i,indices_sauf_i].-=W[i,ind_min]*(SWT[ind_min,indices_sauf_i])
        WSWT[:,i].=WSWT[i,:]
        SWT[:,i].-=S[:,ind_min]*W[i,ind_min]'
        WSWT[i,i]-=2*SWT[ind_min,i]*W[i,ind_min]+W[i,ind_min]^2*S[ind_min,ind_min]
        #maj
        W[i,ind_min]=x_min
        # mise à jour des ind_minré calculs :
        sumW[i]+=W[i,ind_min]
        WSWT[i,i]+=2*SWT[ind_min,i]*W[i,ind_min]+W[i,ind_min]^2*S[ind_min,ind_min]
        SWT[:,i].+=S[:,ind_min]*W[i,ind_min]'
        WSWT[i,indices_sauf_i].+=W[i,ind_min]*(SWT[ind_min,indices_sauf_i])
        WSWT[:,i].=WSWT[i,:]
    end
    return W
end 

function UpdateS(X,W,S,lambda)
    WSWT=W*S*W'
    for k = 1:r
        for l = 1:r
            # supression de l'élément 
            a = 0
            b = 0
            c=0
            ind_k = findall(W[:, k] .> 0)
            ind_l = findall(W[:, l] .> 0)
            for i in ind_k
                for j in ind_l
                    WSWT[i,j]-=(W[i,k]*W[j,l]+W[i,l]*W[j,k])*S[k,l]
                    
                    a += (W[i, k] * W[j, l]+W[j,k]*W[i,l])^2
                
                    b += 2 * (W[i, k] * W[j, l]+W[j,k]*W[i,l])*(WSWT[i,j]-X[i,j])
                
                    
                    c += (X[i,j]-WSWT[i,j])^2
                    
                end
            end
            #mise à jour des pre calculs 
            if a ==0
                S[k,l]=0
            else 
                S[k, l] = min(max(-b / (2a), 0),1)
            end 
            S[l,k]=S[k,l]
            for i in ind_k
                for j in ind_l
                    WSWT[i,j]+=(W[i,k]*W[j,l]+W[i,l]*W[j,k])*S[k,l]
                end 
            end 
           

        end
    end
    return S
end 

function scale_S(W,S)
    Sd = copy(S)
    D = Matrix{Float64}(I, r, r) # Crée une matrice identité r x r
    p = 1
    
    while p == 1 || (minimum(maximum(Sd)) < 0.9999 && p <= 100)
        for i = 1:r
            d = 1 / max(sqrt(Sd[i, i]), maximum(Sd[i, [1:i-1; i+1:r]]))
            Sd[i, :] *= d
            Sd[:, i] *= d
            D[i, i] *= d
        end
        p += 1
    end
    
    Sd, p - 1

    return W*inv(D),D*S*D
end 



function TrisymNMF_CD(X, r,lambda, maxiter,epsi,init_algo="random",time_limit=20)
    debut = time()
    if init_algo=="sspa"
        # Initialisation ONMF avec SSPA calcul de X=WOHO et W=HO'
        
        n = size(X, 1)
        p=max(2,Int(floor(0.1*n/r)))
        options = Dict(:average => 1) # Définissez les options avec lra = 1
        WO,K=SSPA(X, r, p, options)

        norm2x = sqrt.(sum(X.^2, dims=1))
        Xn = X .* (1 ./ (norm2x .+ 1e-16))
        normX2 = sum(X .^ 2)

        e = Float64[]

        HO = orthNNLS(X, WO, Xn)
        W=HO'
        
        #OPtimisation de S
        S=W'*X*W 

        # mise a jour pour que S soit entre 0 et 1
       
        W,S=scale_S(W,S)
       



    end 
    if init_algo=="random"
        # initialisation aléatoire
        n = size(X, 1)
        W = rand(n, r)
        

        matrice_aleatoire = rand(r, r)
        S = 0.5 * (matrice_aleatoire + transpose(matrice_aleatoire))

    end 
    
    erreur_prec = calcul_erreur(X, W, S,lambda)
    erreur = erreur_prec
    println(erreur)
    for itter in 1:maxiter
        # Calculer le temps écoulé
        temps_ecoule = time() - debut

        

        # Vérifier si le temps écoulé est inférieur à la limite
        if temps_ecoule > time_limit
           
            println("Limite de temps dépassée.")
            break
        end
        
        W=UpdateW2(X,W,S,lambda)
        
        S=UpdateS(X,W,S,lambda)
       
        erreur_prec = erreur
        erreur = calcul_erreur(X, W, S,lambda)
       
        if erreur<epsi
            break
        end
        if abs(erreur_prec-erreur)<epsi
            break
        end
    end

    
    erreur = calcul_erreur(X, W, S,lambda)
    
    return W, S, erreur
end
r=5
n=20
W1 = rand(n, r)

# Création de la matrice S symétrique avec des éléments entre 0 et 1
S1 = rand(r, r)

# Rendre S symétrique
S1 = S1 + S1'


X=W1*S1*W1'
lambda=0.3
maxiter=1000
epsi=1e-2
r=3
W, S, erreur=TrisymNMF_CD(X, r,lambda, maxiter,epsi,"sspa")

println(norm(X-W*S*W')/norm(X))
W, S, erreur=TrisymNMF_CD(X, r,lambda, maxiter,epsi,"random")

println(norm(X-W*S*W')/norm(X))
