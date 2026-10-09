# =========================================================================
# Yeo-Johnson-Transformation
# =========================================================================
#
# Varianzstabilisierung für Heteroskedastizität, bei der die Streuung mit dem
# Niveau wächst (siehe Residuen-Diagnostik: Trichter in Residuen vs. Vorhersage).
# Verallgemeinert die Log-Transformation (λ = 0) und funktioniert auch mit
# Nullen und negativen Werten.
#
# Das installierte YeoJohnson.jl-Paket liefert nur die Vorwärts-Transformation
# ohne λ und ohne Inverse; hier deshalb eine vollständige eigene Implementierung:
#
#     yeo_johnson(y, λ)       ...Transformation
#     yeo_johnson_inv(z, λ)   ...Rücktransformation (für Vorhersagen)
#     yeo_johnson_λ(y)        ...λ-Schätzung per Maximum-Likelihood
#     yeo_johnson_spalte!(df, :tges, :yj_tges)  ...vorab berechnete Spalte anlegen
#     yeo_johnson_auf_teil!(teil)               ...dto. direkt auf einem Teil
#
# Workflow (entspricht der Konvention für FORMELN: vorab berechnete Spalten):
#
#     λ̂ = yeo_johnson_auf_teil!(teil)                    # Spalte teil.df.yj_tges
#     mod = theil_sen_regression!(teil, FORMELN[:yeojohnson])
#     y_hat = yeo_johnson_inv(mod.points.y_hat, λ̂)      # zurück auf Sekunden

"""
 # yeo_johnson(y, λ) -> Vector{Float64}

 Yeo-Johnson-Transformation eines Vektors.

 - `y >= 0`: `z = ((y + 1)^λ - 1) / λ` bzw. `log(y + 1)` für λ = 0
 - `y < 0`:  `z = -((1 - y)^(2 - λ) - 1) / (2 - λ)` bzw. `-log(1 - y)` für λ = 2

 λ-Interpretation: λ ≈ 0 verhält sich wie die Log-Transformation,
 λ ≈ 1 wie keine Transformation (nur Scherung), λ < 1 staucht die
 Verteilung und stabilisiert damit niveauabhängige Streuung.
"""
function yeo_johnson(y::AbstractVector{<:Real}, λ::Real)
    return [_yeo_johnson_einzelwert(x, λ) for x in y]
end

function _yeo_johnson_einzelwert(x::Real, λ::Real)
    if x >= 0
        return abs(λ) < 1e-8 ? log1p(x) : ((x + 1)^λ - 1) / λ
    else
        return abs(λ - 2) < 1e-8 ? -log1p(-x) : -((1 - x)^(2 - λ) - 1) / (2 - λ)
    end
end

"""
 # yeo_johnson_inv(z, λ) -> Vector{Float64}

 Inverse Yeo-Johnson-Transformation: rechnet transformierte Werte (z.B.
 Vorhersagen oder Achsenabschnitt/Steigung der transformierten Skala) mit
 demselben λ auf die Originalskala zurück. `z` muss mit demselben λ entstanden
 sein wie in `yeo_johnson`.
"""
function yeo_johnson_inv(z::AbstractVector{<:Real}, λ::Real)
    return [_yeo_johnson_inv_einzelwert(zi, λ) for zi in z]
end

function _yeo_johnson_inv_einzelwert(z::Real, λ::Real)
    if z >= 0
        return abs(λ) < 1e-8 ? expm1(z) : (λ * z + 1)^(1 / λ) - 1
    else
        w = -z
        return abs(λ - 2) < 1e-8 ? 1 - exp(w) : 1 - (1 + w * (2 - λ))^(1 / (2 - λ))
    end
end

"""
 # yeo_johnson_λ(y; lo = -2, hi = 2) -> Float64

 Schätzt λ per Maximum-Likelihood: maximiert die Log-Likelihood der
 normalverteilten transformierten Werte (wie SciPy `yeojohnson_llf`).
 Suche: grober Scan über [lo, hi], danach Golden-Section-Feinsuche.

 Hinweis: Die MLE-Schätzung reagiert auf Ausreißer in `y`. Für saubere
 Ergebnisse λ entweder auf vorgefilterten Zeiten schätzen oder nach dem
 ersten Ausreißer-Durchlauf auf den verbleibenden OK-Punkten erneut schätzen
 und die Spalte mit dem neuen λ neu anlegen.
"""
function yeo_johnson_λ(y::AbstractVector{<:Real}; lo::Real = -2.0, hi::Real = 2.0)
    yv = Float64.(y)
    length(yv) < 2 && throw(ArgumentError("λ-Schätzung benötigt mindestens 2 Werte."))

    # Vorzeichen-Term der Log-Likelihood einmal vorab
    s = sum(sign(xi) * log1p(abs(xi)) for xi in yv)
    loglike(λ) = begin
        σ² = var(yeo_johnson(yv, λ))
        σ² <= 0 && return -Inf
        return -length(yv) / 2 * log(σ²) + (λ - 1) * s
    end

    # Grober Scan, dann Golden-Section-Maximierung im besten Intervall
    grob = collect(lo:0.1:hi)
    λ0 = grob[argmax(loglike.(grob))]
    a = max(lo, λ0 - 0.1)
    b = min(hi, λ0 + 0.1)

    φ = (sqrt(5) - 1) / 2
    c = b - φ * (b - a)
    d = a + φ * (b - a)
    while (b - a) > 1e-6
        if loglike(c) > loglike(d)
            b = d
            d = c
            c = b - φ * (b - a)
        else
            a = c
            c = d
            d = a + φ * (b - a)
        end
    end
    return (a + b) / 2
end

"""
 # yeo_johnson_spalte!(df, quelle, ziel; λ = nothing) -> Float64

 Legt im DataFrame die transformierte Spalte `ziel` an (Missing bleibt Missing)
 und gibt das verwendete λ zurück. Ohne `λ` wird λ per MLE aus der Quellspalte
 geschätzt. Das zurückgegebene λ unbedingt mit abspeichern — es wird für die
 Rücktransformation (`yeo_johnson_inv`) und für neue Daten (z.B. Testdaten in
 der Cross-Validation) benötigt.
"""
function yeo_johnson_spalte!(df::AbstractDataFrame, quelle::Symbol, ziel::Symbol; λ = nothing)
    y = df[!, quelle]
    y_ohne_missing = Float64.(collect(skipmissing(y)))
    λ̂ = isnothing(λ) ? yeo_johnson_λ(y_ohne_missing) : Float64(λ)
    df[!, ziel] = [ismissing(v) ? missing : _yeo_johnson_einzelwert(v, λ̂) for v in y]
    return λ̂
end

"""
 # yeo_johnson_auf_teil!(teil; quelle = :tges, ziel = :yj_tges, λ = nothing) -> Float64

 Convenience-Wrapper für `yeo_johnson_spalte!` auf `teil.df` mit den Standard-
 spalten für die Zeitregression (`:tges` -> `:yj_tges`). Die Zielspalte ist die
 Response der Formel `FORMELN[:yeojohnson]`.
"""
function yeo_johnson_auf_teil!(teil::Teil; quelle::Symbol = :tges, ziel::Symbol = :yj_tges, λ = nothing)
    return yeo_johnson_spalte!(teil.df, quelle, ziel; λ)
end

# =========================================================================
# Interpretation der Koeffizienten auf der Originalskala
# =========================================================================
#
# Das transformierte Modell ist z = n + m * x mit z = YJ(y, λ̂). Weil die
# Yeo-Johnson-Transformation nichtlinear (aber streng monoton) ist, entspricht
# das auf der Originalskala einer KURVE:
#
#     t̂(x) = YJ⁻¹(n + m * x, λ̂)          ...Medianzeit für Auftragsgröße x
#
# - "Achsenabschnitt" = t̂(0) = YJ⁻¹(n, λ̂) ist wohldefiniert (Medianzeit bei
#   rMenge = 0, also im Wesentlichen die Rüstzeit).
# - Eine konstante Steigung in s/Stück existiert NICHT; die Kurve hat an jeder
#   Stelle eine lokale Steigung dy/dx, die mit x wächst (für λ < 1).

"""
 # yeo_johnson_vorhersage(koef, λ, x) -> Vector{Float64}

 Rechnet transformierte Koeffizienten (`koef = [intercept, steigung]`, wie von
 `Theil_Sen_Coefs`/`StatsAPI.coef` geliefert, Intercept zuerst) zusammen mit λ
 in Medianzeiten auf der Originalskala (Sekunden) um:

     t̂(x) = YJ⁻¹(koef[1] + koef[2] * x, λ)

 Da YJ streng monoton ist und Theil-Sen auf der transformierten Skala den
 bedingten Median schätzt, ist t̂(x) die Medianzeit für die Auftragsgröße x —
 verwendbar wie eine Vorhersagefunktion, z.B. zum Einzeichnen der Kurve in
 den Streudiagramm.
"""
function yeo_johnson_vorhersage(koef::AbstractVector{<:Real}, λ::Real, x::AbstractVector{<:Real})
    length(koef) == 2 || throw(ArgumentError(
        "koef muss [intercept, steigung] enthalten, erhalten: $(koef)"))
    return yeo_johnson_inv(koef[1] .+ koef[2] .* Float64.(x), λ)
end

"""
 # yeo_johnson_achsenabschnitt(koef, λ) -> Float64

 Medianzeit bei rMenge = 0, also der zurücktransformierte Achsenabschnitt:
 `YJ⁻¹(koef[1], λ)`. Das ist das Originalskalen-Pendant zu `n` im stats-DataFrame.
"""
yeo_johnson_achsenabschnitt(koef::AbstractVector{<:Real}, λ::Real) =
    only(yeo_johnson_vorhersage(koef, λ, [0.0]))

# Ableitung der inversen Transformation nach z (Kettenregel: dy/dx = m * dy/dz)
function _yeo_johnson_inv_ableitung(z::Real, λ::Real)
    if z >= 0
        return abs(λ) < 1e-8 ? exp(z) : (λ * z + 1)^(1 / λ - 1)
    else
        w = -z
        return abs(λ - 2) < 1e-8 ? exp(w) : (1 + w * (2 - λ))^(1 / (2 - λ) - 1)
    end
end

"""
 # yeo_johnson_steigungen(koef, λ, x) -> Vector{Float64}

 Lokale Steigung der zurücktransformierten Kurve an den Stellen x, also
 Sekunden je zusätzlichem Stück bei der jeweiligen Auftragsgröße:

     dy/dx = m * dYJ⁻¹/dz (n + m * x, λ)

 Für λ < 1 wächst diese lokale Steigung mit x — genau das macht die Kurve auf
 der Originalskala aus. Für eine einzige aussagekräftige Zahl bietet sich die
 lokale Steigung an der medianen Auftragsgröße an.
"""
# Skalar: ein Punkt
function yeo_johnson_steigungen(koef::AbstractVector{<:Real}, λ::Real, x::Real)
    length(koef) == 2 || throw(ArgumentError(
        "koef muss [intercept, steigung] enthalten, erhalten: $(koef)"))
    z = koef[1] + koef[2] * Float64(x)
    return koef[2] * _yeo_johnson_inv_ableitung(z, λ)
end

# Vektor: delegiert an die Skalar-Methode
function yeo_johnson_steigungen(koef::AbstractVector{<:Real}, λ::Real, x::AbstractVector{<:Real})
    return [yeo_johnson_steigungen(koef, λ, xi) for xi in x]
end
