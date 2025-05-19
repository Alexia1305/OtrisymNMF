"""
    Comparison of initialization for different noise levels on synthetic data
"""

include("../algo/OtrisymNMF.jl")
using DelimitedFiles, Plots, SparseArrays, LaTeXStrings, Random, LinearAlgebra

Random.seed!(2001)
gr()  # Configure the GR backend

# Global parameters
nbr_level = 5
n = 200
const rin = 5
nbr_test = 10
maxiter = 10000
epsi = 1e-5
epsilon = collect(range(0, stop=1, length=nbr_level))

# Result matrices
results = [zeros(nbr_level, 4) for _ in 1:4]

# Function to generate the W_true matrix
function generate_W_true(n, r)
    W = zeros(n, r)
    for i in 1:n
        W[i, rand(1:r)] = rand() + 1
    end
    W[:, findall(x -> any(x .!= 0), eachcol(W))]  # Remove zero columns
end

# Normalize the columns of a matrix
function normalize_columns!(W)
    for j in 1:size(W, 2)
        W[:, j] ./= norm(W[:, j], 2)
    end
end

# Generate the similarity matrix S
function generate_S(r)
    S = Matrix(sprand(r, r, 0.3))
    S = 0.5 * (S + transpose(S))
    for k in 1:r
        S[k, k] = 1
    end
    return S
end

# Add noise to matrix X
function add_noise(X, level)
    if epsilon[level] != 0
        N = randn(n, n) * epsilon[level] * norm(X) / norm(randn(n, n))
        X = max.(X + N, 0)  # No negative values
    end
    return X
end

# Main function that runs the tests
function run_experiment()
    algos = ["k_means", "sspa", "random", "spa"]

    for level in 1:nbr_level
        println("Noise level: $level")
        r = rin
        stats = zeros(4, 4)  # Matrix to store statistics (1 row per algorithm)

        for test in 1:nbr_test
            W_true = generate_W_true(n, r)
            r = size(W_true, 2)
            normalize_columns!(W_true)

            X = W_true * generate_S(r) * transpose(W_true)
            X = add_noise(X, level)

            # Run each algorithm
            for i in 1:4
                time = @elapsed W, S, err = OtrisymNMF_CD(X, r, maxiter, epsi, algos[i])
                acc = calcul_accuracy(W_true, W)
                stats[i, :] .+= [acc * 100, time, err, (acc == 1 ? 1 : 0) * 100] ./ nbr_test
            end
        end

        for i in 1:4
            results[i][level, :] = stats[i, :]
        end
    end
end

function plot_results(epsilon, results, ylabel, filename; metric_index=1)
    labels = ["init kmeans", "init sspa", "init random", "init spa"]
    colors = [:blue, :red, :green, :purple]
    linestyles = [:solid, :dash, :dot, :dashdot]

    p = plot(title=ylabel, xlabel="Noise level (ε)", ylabel=ylabel, legend=:bottomleft)

    for i in 1:4
        y = results[i][:, metric_index]
        plot!(p, epsilon, y, label=labels[i], linecolor=colors[i], linestyle=linestyles[i], linewidth=2)
        scatter!(p, epsilon, y, markercolor=colors[i],label="")
    end

    savefig(p, filename)
end

function save_results(filename, epsilon, results)
    open(filename, "w") do f
        for (name, res) in zip(["kmeans", "sspa", "random", "spa"], results)
            write(f, "$name:\n")
            for row in eachrow(res)
                write(f, "$(collect(row))\n")
            end
        end
    end
end

# Exécution
run_experiment()

# Figures
plot_results(epsilon, results, "Success rate (%)", "success_rate.png", metric_index=4)
plot_results(epsilon, results, "Accuracy (%)", "accuracy.png", metric_index=1)
plot_results(epsilon, results, "Time (s)", "time.png", metric_index=2)
plot_results(epsilon, results, "Relative error", "rel_error.png", metric_index=3)

# Sauvegarde
save_results("resultats_initssed.txt", epsilon, results)