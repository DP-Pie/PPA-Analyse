# Muss noch richtig implementiert werden, ersialisation und soweiter.
#@save "Teiledict.jld2" Teiledict
#@load "Teiledict.jld2" Teiledict
function theil_sen_regression!(teil::Teil)

    daten = dropmissing(
        select(teil.df, [:rMenge, :tges]),
        [:rMenge, :tges]
    )

    x = Float64.(daten.rMenge)
    y = Float64.(daten.tges)

    len = length(x)

    # Überprüfen, ob die Vektoren die gleiche Länge haben
    len != length(y) ? throw(ArgumentError("Die Vektoren x und y müssen die gleiche Länge haben.")) : nothing
    # Überprüfen, ob die Vektoren mindestens zwei Elemente enthalten
    len < 2 ? throw(ArgumentError("Die Vektoren x und y müssen mindestens zwei Elemente enthalten.")) : nothing

    @inline m(x1::Float64, x2::Float64, y1::Float64, y2::Float64) = (y1 - y2) / (x1 - x2)

    slopes = Float64[]

    for i in 1:(len - 1)
        for j in (i + 1):len
            if x[i] != x[j]
                slope = m(x[i], x[j], y[i], y[j])
                push!(slopes, slope)
            end
        end
    end

    # Überprüft slopes auf leeren Inhalt.
    isempty(slopes) ? throw(ArgumentError("Es gibt keine unterschiedlichen x-Werte.")) : nothing

    slope = median(slopes)

    intercepts = y .- slope .* x
    intercept = median(intercepts)

    y_hat = slope .* x .+ intercept
    e = y .- y_hat

    # Regressionsmodelldaten zusammenstellen
    stats_df = DataFrame(
        Regression = ["Theil-Sen"],
        LOSS_Fkt = ["Median"],
        m = [slope],
        n = [intercept],
        std_err = [NaN],
        r2 = [NaN]
        )
    points_df = DataFrame(
        x = x,
        y = y,
        y_hat = y_hat,
        residuals = e
    )
    modell = RegressionsModell(
        name = "Theil-Sen",
        stats = stats_df,
        points = points_df
    )

    teil.RegressionsModell[modell.name] = modell

    return modell
end

function Theil_Sen_Core(x::Vector{Float64},y::Vector{Float64})
    @inline m(x1::Float64, x2::Float64, y1::Float64, y2::Float64) = (y1 - y2) / (x1 - x2)

    len = length(x)

    # Überprüfen, ob die Vektoren die gleiche Länge haben
    len != length(y) ? throw(ArgumentError("Die Vektoren x und y müssen die gleiche Länge haben.")) : nothing
    # Überprüfen, ob die Vektoren mindestens zwei Elemente enthalten
    len < 2 ? throw(ArgumentError("Die Vektoren x und y müssen mindestens zwei Elemente enthalten.")) : nothing

    slopes = Float64[]

    for i in 1:(len - 1)
        for j in (i + 1):len
            if x[i] != x[j]
                slope = m(x[i], x[j], y[i], y[j])
                push!(slopes, slope)
            end
        end
    end

    # Überprüft slopes auf leeren Inhalt.
    isempty(slopes) ? throw(ArgumentError("Es gibt keine unterschiedlichen x-Werte.")) : nothing

    slope = median(slopes)

    intercepts = y .- slope .* x
    intercept = median(intercepts)

    y_hat = slope .* x .+ intercept
    e = y .- y_hat
    return slope, intercept, e
end

# Robuste Regression mit MM-Estimator
function mm_regression!( teil::Teil; min_n::Int = 10 )

    daten = dropmissing(
        select(teil.df, [:rMenge, :tges]),
        [:rMenge, :tges]
    )

    if nrow(daten) <= min_n
        return nothing
    end

    x = Float64.(daten.rMenge)
    y = Float64.(daten.tges)

    if length(unique(x)) < 2
        throw(ArgumentError(
            "Zu wenige unterschiedliche x-Werte für Teil $(teil.name)."
        ))
    end

    regressionsdaten = DataFrame(
        x = x,
        y = y
    )

    modell_robust = RobustModels.rlm(
        @formula(y ~ x),
        regressionsdaten,
        RobustModels.MMEstimator{RobustModels.TukeyLoss}();
        σ0 = :mad
    )

    koeffizienten = StatsAPI.coef(modell_robust)
    standardfehler = StatsAPI.stderror(modell_robust)

    intercept = koeffizienten[1]
    slope = koeffizienten[2]

    y_hat = slope .* x .+ intercept
    residuals = y .- y_hat

    summe_quadrate = sum((y .- mean(y)) .^ 2)

    r2 = if summe_quadrate == 0
        NaN
    else
        1 - sum(residuals .^ 2) / summe_quadrate
    end

    stats_df = DataFrame(
        Regression = ["MM"],
        LOSS_Fkt = ["TukeyLoss"],
        m = [slope],
        std_err_m = [standardfehler[2]],
        std_err_intercept = [standardfehler[1]],
        n = [intercept],
        r2 = [r2]
    )

    points_df = DataFrame(
        x = x,
        y = y,
        y_hat = y_hat,
        residuals = residuals
    )

    modell = RegressionsModell(
        name = "MM-Tukey",
        stats = stats_df,
        points = points_df
    )

    teil.RegressionsModell[modell.name] = modell

    return modell
end