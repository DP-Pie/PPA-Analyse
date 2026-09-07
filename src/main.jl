using Pkg

REQUIRED_PACKAGES = [
    "NativeFileDialog",
    "RobustModels",
    "LinRegOutliers",
    "DataFrames",
    "CSV",
    "GLMakie",
    "JLD2",
    "CairoMakie",
    "Statistics",
    "GLM",
    "StatsModels",
    "StatsAPI"
]

# Installiere fehlende Pakete
for package in REQUIRED_PACKAGES
    Base.find_package(package) === nothing && Pkg.add(package)
end

using NativeFileDialog
using RobustModels
using LinRegOutliers
using DataFrames
using CSV
using GLMakie
using JLD2
using CairoMakie
using Statistics
using GLM
using StatsModels
using StatsAPI

include("Konstructs.jl")

df_path = pick_file(filterlist="*csv")
df = CSV.read(df_path, DataFrame)

Teiledict = erstelle_teile(df)

# Diagramme anzeigen
include("Diagrams.jl")
# Anzahl der Einträge je Teil, absteigend sortiert
Show_einträge()

# Speichern der Eintragsverteilung als SVG-Datei
CairoMakie.activate!()
save("eintragsverteilung.svg", fig)
vscodedisplay(df)

# Muss noch richtig implementiert werden, ersialisation und soweiter.
#@save "Teiledict.jld2" Teiledict
#@load "Teiledict.jld2" Teiledict
function theil_sen_regression(
    x::Vector{Int64},
    y::Vector{Float64}
)
    x = Float64.(x)
    return theil_sen_regression(x, y)
end

function theil_sen_regression(
    x::Vector{Float64},
    y::Vector{Float64}
)
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

    return modell
end


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
        intercept = [intercept],
        std_err_intercept = [standardfehler[1]],
        n = [length(x)],
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

    #teil.RegressionsModell["MM-Tukey"] = modell

    return modell
end

# Theil-Sen Regression für jedes Teil durchführen, das mindestens 20 Einträge hat
for (key,Teil) in Teiledict
    if Teil.n >= 100
        #println("$key hat $(Teil.n) Einträge.")
        MyReg = theil_sen_regression(Teil.df.rMenge, Teil.df.tges)
        println("Theil-Sen Regression für Teil $key: m = $(MyReg.stats.m[1]), n = $(MyReg.stats.n[1])")
        Teil.RegressionsModell["Theil-Sen"] = MyReg
        MyReg = mm_regression(Teil)
        println("Theil-Sen Regression für Teil $key: m = $(MyReg.stats.m[1]), n = $(MyReg.stats.n[1])")
        Teil.RegressionsModell["MM-Tukey"] = MyReg
    end
end 

vscodedisplay(Teiledict["KK-SK343-75-AA-002"].df)
Teiledict["KK-SK343-75-AA-002"].df.rMenge |> typeof
liste_regressionsmodelle(Teiledict["KK-SK343-75-AA-002"])