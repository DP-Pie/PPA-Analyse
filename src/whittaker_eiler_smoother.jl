#Info: https://medium.com/@anbowell
using SparseArrays
using LinearAlgebra

# =============================================
# 1. Optimierte difference_matrix (gibt (n-d) × n sparse Matrix zurück)
# =============================================
function difference_matrix(n::Int, d::Int=2)
    if d <= 0
        return Matrix{Float64}(I, n, n)  # Fallback für d ≤ 0
    end
    # Starte mit n × n sparse Einheitsmatrix
    D = spdiagm(0 => ones(n))
    for _ in 1:d
        D = diff(D, dims=1)  # diff funktioniert mit sparse Matrizen!
    end
    # Ergebnis: (n-d) × n sparse Matrix
    return D
end

# =============================================
# 2. Optimierter whittaker_smoother
# =============================================
function whittaker_smoother(y::Vector{Float64}, λ::Float64, d::Int=2)
    n = length(y)
    if d == 0
        return copy(y)  # λ * D'D = 0 → z = y
    end
    # Erstelle D als (n-d) × n sparse Matrix
    D = spdiagm(0 => ones(n))  # n × n sparse
    for _ in 1:d
        D = diff(D, dims=1)  # (n-d) × n nach d Iterationen
    end
    DtD = D' * D  # n × n sparse Matrix (D'D)
    W = I + λ * DtD  # n × n sparse (W = I + λ·D'D)
    z = W \ y  # Löst W * z = y (sparse Backslash nutzt Cholesky für SPD-Matrizen)
    return z
end