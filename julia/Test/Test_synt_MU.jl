include("../algo/OtrisymNMF.jl")
using DelimitedFiles
using Plots
gr()  # Configure the GR backend
using SparseArrays
using LaTeXStrings
using Random

Random.seed!(2001)

# Parameters
nbr_level = 5
n = 200
const rin = 8
epsilon = collect(range(0, stop=1, length=nbr_level))

# Initialize results matrices
result = zeros(length(epsilon), 4)
result2 = zeros(length(epsilon), 4)

# Iterate over noise levels
for level in 1:nbr_level
    r = rin
    println("Noise level: $level")
    
    nbr_test = 10
    accuracy_moy = 0.0
    accuracy_moy2 = 0.0
    time1 = 0.0
    time2 = 0.0
    error1 = 0.0
    error2 = 0.0
    succes = 0
    succes2 = 0

    # Run tests for each noise level
    for test in 1:nbr_test
        println("Test number: $test")
        
        # Generate the true W matrix (W_true)
        W_true2 = zeros(n, r)
        for i in 1:n
            k = rand(1:r)
            W_true2[i, k] = rand() + 1
        end
        
        # Remove zero columns
        non_zero_columns = findall(x -> any(x .!= 0), eachcol(W_true2))
        W_true = W_true2[:, non_zero_columns]
        r = size(W_true, 2)

        # Normalize W_true columns
        for j in 1:r
            W_true2[:, j] .= W_true2[:, j] ./ norm(W_true2[:, j], 2)
        end

        # Density of the matrix (proportion of non-zero elements)
        density = 0.3
        
        # Generate a random sparse matrix
        random_sparse_matrix = sprand(r, r, density)
        S = Matrix(random_sparse_matrix)
        S = 0.5 * (S + transpose(S))
        
        # Set diagonal elements to 1
        for k in 1:r
            S[k, k] = 1
        end
        
        # Generate the noisy matrix X
        X = W_true * S * transpose(W_true)

        # Add noise if applicable
        if epsilon[level] != 0 
            N = randn(n, n)
            N = epsilon[level] * (N / norm(N)) * norm(X)
            X = X + N
            X = max.(X, 0)  # No negative values
        end
        
        # Set maximum iterations and epsilon for algorithms
        maxiter = 10000
        epsi = 1e-5

        # Run the OtrisymNMF_CD algorithm
        temps_execution_1 = @elapsed begin
            W, S, erreur = OtrisymNMF_CD(X, r, maxiter, epsi, "sspa")
        end
        
        # Run the OtrisymNMF_MU algorithm
        temps_execution_2 = @elapsed begin
            W2, S2, erreur2 = OtrisymNMF_MU(X, r, maxiter, epsi, "sspa")
        end
        
        # Calculate accuracy for both algorithms
        accu1 = calcul_accuracy(W_true, W)
        accu2 = calcul_accuracy(W_true, W2)

        # Aggregate the results
        accuracy_moy += accu1
        accuracy_moy2 += accu2
        if accu1 == 1
            succes += 1
        end
        if accu2 == 1
            succes2 += 1
        end
        time1 += temps_execution_1
        time2 += temps_execution_2
        error1 += erreur
        error2 += erreur2
    end 
    
    # Average the results over all tests
    accuracy_moy /= nbr_test / 100
    accuracy_moy2 /= nbr_test / 100
    time1 /= nbr_test
    time2 /= nbr_test
    error1 /= nbr_test
    error2 /= nbr_test

    # Store the results
    println("Accuracy CD: $accuracy_moy")
    println("Accuracy MU: $accuracy_moy2")
    
    result[level, 1] = accuracy_moy
    result2[level, 1] = accuracy_moy2
    result[level, 2] = time1
    result2[level, 2] = time2
    result[level, 3] = error1
    result2[level, 3] = error2
    result[level, 4] = succes * 100 / nbr_test
    result2[level, 4] = succes2 * 100 / nbr_test
end 

# Set font size for plots
font_size = 14
plot_font = "Computer Modern"
default(
    fontfamily=plot_font,
    guidefontsize=font_size,
    linewidth=2, 
    framestyle=:box, 
    label=nothing, 
    grid=false
)

# Plot success rate
plot(epsilon, result[:, 4], label="CD", xlabel="epsilon", ylabel="Success rate (%)", xtickfont=font_size, ytickfont=font_size, legendfont=font_size, linecolor=:blue)
scatter!(epsilon, result[:, 4], label="", markercolor=:blue)
plot!(epsilon, result2[:, 4], label="MU", linestyle=:dash, linecolor=:red)
scatter!(epsilon, result2[:, 4], label="", markercolor=:red)
savefig("figure4.png")

# Plot accuracy
plot(epsilon, result[:, 1], label="CD", xlabel="epsilon", ylabel="Accuracy (%)", xtickfont=font_size, ytickfont=font_size, legendfont=font_size, linecolor=:blue)
scatter!(epsilon, result[:, 1], label="", markercolor=:blue)
plot!(epsilon, result2[:, 1], label="MU", linecolor=:red, linestyle=:dash)
scatter!(epsilon, result2[:, 1], label="", markercolor=:red)
savefig("figure.png")

# Plot time
plot(epsilon, result[:, 2], label="CD", xlabel="epsilon", ylabel="Time (s)", xtickfont=font_size, ytickfont=font_size, legendfont=font_size, linecolor=:blue)
scatter!(epsilon, result[:, 2], label="", markercolor=:blue)
plot!(epsilon, result2[:, 2], label="MU", linecolor=:red, linestyle=:dash)
scatter!(epsilon, result2[:, 2], label="", markercolor=:red)
savefig("figure2.png")

# Plot relative error
plot(epsilon, result[:, 3], label="CD", xlabel="epsilon", ylabel="Relative error", xtickfont=font_size, ytickfont=font_size, legendfont=font_size, ylim=(0, :auto), linecolor=:blue)
scatter!(epsilon, result[:, 3], label="", markercolor=:blue)
plot!(epsilon, result2[:, 3], label="MU", linecolor=:red, linestyle=:dash)
scatter!(epsilon, result2[:, 3], label="", markercolor=:red)
savefig("figure3.png")

# Save results to a text file
file_name = "CDvsMU.txt"
open(file_name, "w") do f
    write(f, "$(epsilon)\n")
    write(f, "CD Results:\n")
    for row in eachrow(result)
        write(f, "$(row)\n")
    end 
    write(f, "MU Results:\n")
    for row in eachrow(result2)
        write(f, "$(row)\n")
    end 
end
