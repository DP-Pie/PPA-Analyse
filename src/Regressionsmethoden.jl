# =========================================================================
# Datenvorbereitung
# =========================================================================
function PrepareData!(teil::Teil, Reg::Symbol = :MM, formel = FORMELN[:linear];
        t_rstudent::Float64 = 2.5,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    ausreißer_flags_setzen!(teil, Reg, formel;
    t_rstudent,
    hebel_faktor,
    t_cooks,
    ukz_n,
    ukz_tges,
    verdacht_min
    )

    Messpunkte_df = filter(:Ausreißer => a -> a != "Ausreißer", teil.df)
    n_Messpunkte = nrow(Messpunkte_df)
    n_Ausreißer = nrow(teil.df) - n_Messpunkte
    Ausreißer_pro = n_Ausreißer/nrow(teil.df)

    Spalten_Auswahl = StatsModels.termvars(formel)
    Rückgabe_df = select(Messpunkte_df, Spalten_Auswahl)

    return Rückgabe_df, n_Messpunkte, n_Ausreißer, Ausreißer_pro
end

# =========================================================================
# Theil Sen Regression
# =========================================================================
function theil_sen_regression!(teil::Teil, formel = FORMELN[:linear];
        t_rstudent::Float64 = 2.5,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    OK_df, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(teil, :TheilSen, formel;
        t_rstudent,
        hebel_faktor,
        t_cooks,
        ukz_n,
        ukz_tges,
        verdacht_min
    )

    ts_modell = TheilSenModell(formel, OK_df)
    intercept, slope = ts_modell.coef

    y = Float64.(OK_df[!, ts_modell.yvar])
    ŷ = StatsAPI.predict(ts_modell, OK_df)
    e = y .- ŷ

    # Regressionsmodelldaten zusammenstellen
    stats_df = DataFrame(
        Regression = ["Theil-Sen"],
        LOSS_Fkt = ["Median"],
        n_Rohdaten = teil.n,
        n_OK = Messpunkte,
        n_Ausreißer = Ausreißer,
        pro_Ausreißer = Ausreißer_pro,
        m = [slope],
        n = [intercept],
        std_err = [NaN],
        r2 = [NaN]
        )

    points_df = DataFrame(
        x = Float64.(OK_df[!, ts_modell.xvar]),
        y = y,
        y_hat = ŷ,
        residuals = e
    )

    modell = RegressionsModell(
        name = "Theil-Sen",
        modell = ts_modell,
        stats = stats_df,
        points = points_df
    )

    teil.RegressionsModell[modell.name] = modell

    return modell
end

"""
 # theil_sen_regression_yj!(teil; kwargs...) -> NamedTuple

 Kombinierter Workflow „YJ bereinigt, linear berichtet":

 1. λ̂ auf den Rohzeiten schätzen und `teil.df.yj_tges` anlegen
    (`yeo_johnson_auf_teil!`, Pass 1).
 2. Ausreißer-Flags auf der YJ-Skala bestimmen (`PrepareData!` mit
    `FORMELN[:yeojohnson]`): Auf der stabilisierten Skala gelten
    `t_rstudent`/`hebel`/`cooks` für alle Auftragsgrößen gleichmäßig
    (kein Trichter-Effekt in der Klassifikation).
 3. λ̂ auf den verbleibenden OK-Punkten NEU schätzen (die MLE-Schätzung ist
    ausreißerempfindlich, Pass 2) und die YJ-Spalte mit dem finalen λ̂
    neu anlegen.
 4. Zwei Modelle auf denselben bereinigten Punkten fitten und im
    `RegressionsModell`-Dict des Teils ablegen:
    - `"Theil-Sen"`      ... lineare Originalskala: Rüstzeit/Bauteilzeit in
      Sekunden, Interpretationsmodell.
    - `"Theil-Sen YJ"`   ... YJ-Skala: relative Aussagen und Prozentbänder;
      λ̂ steht in `stats.λ` für Rücktransformationen bereit
      (`yeo_johnson_inv`, `yeo_johnson_vorhersage`, `yeo_johnson_steigungen`).

 ## Rückgabe

 Named Tuple `(lin, yj, λ)`: lineares `RegressionsModell`, YJ-`RegressionsModell`
 und das finale λ̂.

 ## Beispiel
 ```julia
 erg = theil_sen_regression_yj!(Teiledict["KK-SK343-75-AA-002"])
 erg.lin.stats.m        # Bauteilzeit in s/Stück (Originalskala)
 erg.yj.stats.λ[1]      # λ̂ für Rücktransformationen
 erg.λ                  # dto., direkt
 ```
"""
function theil_sen_regression_yj!(teil::Teil;
        t_rstudent::Float64 = 2.5,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    # --- Pass 1: λ̂ auf Rohdaten, Flags auf der YJ-Skala -------------------
    yeo_johnson_auf_teil!(teil)                       # λ̂-Pass 1, Spalte :yj_tges
    _, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(
        teil, :TheilSen, FORMELN[:yeojohnson];
        t_rstudent = t_rstudent,
        hebel_faktor = hebel_faktor,
        t_cooks = t_cooks,
        ukz_n = ukz_n,
        ukz_tges = ukz_tges,
        verdacht_min = verdacht_min
    )

    # Bereinigte Punkte mit den Originalspalten; die Zeilenauswahl stammt aus
    # den YJ-Flags. (Kein erneutes PrepareData! mit der linearen Formel — das
    # würde die Flags auf der linearen Skala neu berechnen.)
    OK_df = dropmissing(select(
        filter(:Ausreißer => a -> a != "Ausreißer", teil.df),
        [:tges, :rMenge]
    ))

    # --- Pass 2: finales λ̂ auf den OK-Punkten, YJ-Werte final --------------
    λ̂ = yeo_johnson_λ(Float64.(OK_df.tges))
    yeo_johnson_auf_teil!(teil; λ = λ̂)               # Spalte mit finalem λ̂
    OK_yj_df = DataFrame(
        rMenge = OK_df.rMenge,
        yj_tges = yeo_johnson(Float64.(OK_df.tges), λ̂)
    )

    # --- Modell 1: lineare Originalskala (Interpretation) ------------------
    ts_lin = TheilSenModell(FORMELN[:linear], OK_df)
    y = Float64.(OK_df[!, ts_lin.yvar])
    ŷ = StatsAPI.predict(ts_lin, OK_df)

    stats_lin = DataFrame(
        Regression = ["Theil-Sen"],
        LOSS_Fkt = ["Median"],
        n_Rohdaten = teil.n,
        n_OK = Messpunkte,
        n_Ausreißer = Ausreißer,
        pro_Ausreißer = Ausreißer_pro,
        m = [ts_lin.coef[2]],
        n = [ts_lin.coef[1]],
        std_err = [NaN],
        r2 = [NaN]
    )

    points_lin = DataFrame(
        x = Float64.(OK_df[!, ts_lin.xvar]),
        y = y,
        y_hat = ŷ,
        residuals = y .- ŷ
    )

    modell_lin = RegressionsModell(
        name = "Theil-Sen",
        modell = ts_lin,
        stats = stats_lin,
        points = points_lin
    )

    # --- Modell 2: YJ-Skala (relative Sicht) --------------------------------
    ts_yj = TheilSenModell(FORMELN[:yeojohnson], OK_yj_df)
    z = Float64.(OK_yj_df[!, ts_yj.yvar])
    ẑ = StatsAPI.predict(ts_yj, OK_yj_df)

    stats_yj = DataFrame(
        Regression = ["Theil-Sen YJ"],
        LOSS_Fkt = ["Median"],
        n_Rohdaten = teil.n,
        n_OK = Messpunkte,
        n_Ausreißer = Ausreißer,
        pro_Ausreißer = Ausreißer_pro,
        λ = [λ̂],                    # für Rücktransformationen mitabgelegt
        m = [ts_yj.coef[2]],
        n = [ts_yj.coef[1]],
        std_err = [NaN],
        r2 = [NaN]
    )

    points_yj = DataFrame(
        x = Float64.(OK_yj_df[!, ts_yj.xvar]),
        y = z,
        y_hat = ẑ,
        residuals = z .- ẑ
    )

    modell_yj = RegressionsModell(
        name = "Theil-Sen YJ",
        modell = ts_yj,
        stats = stats_yj,
        points = points_yj
    )

    teil.RegressionsModell[modell_lin.name] = modell_lin
    teil.RegressionsModell[modell_yj.name] = modell_yj

    return (lin = modell_lin, yj = modell_yj, λ = λ̂)
end

"""
 # Theil_Sen_Coefs(x::Vector{Float64}, y::Vector{Float64}) -> (m, n)

 Bootstrap-Kern der Theil-Sen-Regression: Steigung `m` und Achsenabschnitt `n`
 als Median der paarweisen Steigungen bzw. Achsenabschnitte. Anders als
 `Theil_Sen_Core` werden `y_hat` und Residuen nicht berechnet, damit in der
 Bootstrap-Schleife keine unnötigen Vektoren im Speicher zugewiesen werden.

 Komplexität: O(n²) pro Aufruf.

 ## Fehler

 - `ArgumentError`, wenn x und y unterschiedlich lang oder kürzer als 2
   Elemente sind, oder wenn alle x-Werte identisch sind.
"""
function Theil_Sen_Coefs(x::Vector{Float64}, y::Vector{Float64})
    len = length(x)

    len != length(y) && throw(ArgumentError("Die Vektoren x und y müssen die gleiche Länge haben."))
    len < 2 && throw(ArgumentError("Die Vektoren x und y müssen mindestens zwei Elemente enthalten."))

    # Steigungen aller Punktpaare; vorab alloziert statt push!
    slopes = Vector{Float64}(undef, len * (len - 1) ÷ 2)
    p = 0
    @inbounds for i in 1:(len - 1)
        for j in (i + 1):len
            if x[i] != x[j]
                slopes[p += 1] = (y[i] - y[j]) / (x[i] - x[j])
            end
        end
    end
    p == 0 && throw(ArgumentError("Es gibt keine unterschiedlichen x-Werte."))

    m = median(view(slopes, 1:p))
    n = median(y .- m .* x)
    return m, n
end

"""
 # Theil_Sen_Coefs(f::StatsModels.FormulaTerm, daten::DataFrame) -> Vector{Float64}

 Formel-Variante von `Theil_Sen_Coefs` für das einheitliche Core-Interface.
 Erlaubt sind nur Formeln der Art `y ~ x` (genau ein Prädiktor).

 ## Rückgabe

 `[Achsenabschnitt, Steigung]` - Intercept zuerst, wie bei StatsModels, damit
 die Spalten des Bootstrap-Ergebnisses mit den Formel-Termnamen zusammenpassen.
"""
function Theil_Sen_Coefs(f::StatsModels.FormulaTerm, daten::DataFrame)
    yvars = collect(StatsModels.termvars(f.lhs))
    xvars = collect(StatsModels.termvars(f.rhs))
    (length(yvars) == 1 && length(xvars) == 1) || throw(ArgumentError(
        "Theil-Sen unterstützt nur Formeln der Art y ~ x (erhalten: $(f))"))

    x = Float64.(daten[!, xvars[1]])
    y = Float64.(daten[!, yvars[1]])
    m, n = Theil_Sen_Coefs(x, y)
    return [n, m]
end

# =========================================================================
# MM-Regression
# =========================================================================

# Robuste Regression mit MM-Estimator
#//////////////////////////////////////////////////
#todo
# PosDefException FallBack nun doch Implementieren
#//////////////////////////////////////////////////
function mm_regression!(teil::Teil, formel = FORMELN[:linear];
        t_rstudent::Float64 = 2.5,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2,
        maxiter::Integer = 1000)

    OK_df, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(teil, :MM, formel;
        t_rstudent,
        hebel_faktor,
        t_cooks,
        ukz_n,
        ukz_tges,
        verdacht_min
    )

    modell_robust = mm_regression_fit(formel, OK_df; maxiter = maxiter)

    koeffizienten = StatsAPI.coef(modell_robust)
    standardfehler = StatsAPI.stderror(modell_robust)
    # Termnamen passend zu den Koeffizienten, z.B. ["(Intercept)", "rMenge", "rMenge & Länge"]
    termnamen = string.(StatsAPI.coefnames(modell_robust))

    # Ziel- und Prädiktorvariablen aus der Formel statt fester Spaltenindizes
    yvar = only(collect(StatsModels.termvars(formel.lhs)))
    xvars = collect(StatsModels.termvars(formel.rhs))

    y = Float64.(OK_df[!, yvar])
    y_hat = StatsAPI.predict(modell_robust)   # Fitted Values des ganzen Modells, beliebige Anzahl Terme
    residuals = y .- y_hat    # nicht StatsAPI.residuals: liefert die IRLS-Arbeitsresiduen

    summe_quadrate = sum((y .- mean(y)) .^ 2)

    r2 = if summe_quadrate == 0
        NaN
    else
        1 - sum(residuals .^ 2) / summe_quadrate
    end

    # Long-Format: eine Zeile je Term/Koeffizient; modellweite Kennzahlen je Zeile wiederholt
    stats_df = DataFrame(
        Regression = fill("MM", length(termnamen)),
        LOSS_Fkt = fill("TukeyLoss", length(termnamen)),
        Term = termnamen,
        n_Rohdaten = fill(teil.n, length(termnamen)),
        n_OK = fill(Messpunkte, length(termnamen)),
        n_Ausreißer = fill(Ausreißer, length(termnamen)),
        pro_Ausreißer = fill(Ausreißer_pro, length(termnamen)),
        coef = koeffizienten,
        std_err = standardfehler,
        r2 = fill(r2, length(termnamen))
    )

    # Je Prädiktor eine Spalte unter dem Originalnamen; bei genau einem Prädiktor
    # heißt sie :x, damit Auswertungen über points.x weiterlaufen
    points_df = select(OK_df, xvars)
    length(xvars) == 1 && rename!(points_df, xvars[1] => :x)
    points_df[!, :y] = y
    points_df[!, :y_hat] = y_hat
    points_df[!, :residuals] = residuals

    modell = RegressionsModell(
        name = "MM-Tukey",
        modell = modell_robust,
        stats = stats_df,
        points = points_df
    )

    teil.RegressionsModell[modell.name] = modell

    return modell
end

"""
 # MM_Core(f::StatsModels.FormulaTerm, daten::DataFrame; estimator=...) -> Vector{Float64}

 Bootstrap-Kern der robusten MM-Regression (`RobustModels.rlm`). Die Formel ist
 Parameter, damit auch Kurven mit mehreren Termen verwendet werden können,
 z.B. `@formula(tges ~ rMenge + rMenge & Länge)`.

 ## Rückgabe

 Koeffizientenvektor (`StatsAPI.coef`), Intercept zuerst.

 ## Argumente

 - `estimator`: RobustModels-Estimator-Typ, Standard
   `RobustModels.MMEstimator{RobustModels.TukeyLoss}`; alternativ z.B.
   `RobustModels.MEstimator{RobustModels.HuberLoss}`.
"""
function MM_Core(f::StatsModels.FormulaTerm, daten::DataFrame;
    # Bei fehlschlag eventuell Fallback auf anderen Schätzer setzen. 
    estimator::Type = RobustModels.MMEstimator{RobustModels.TukeyLoss},
    ReturnModell::Bool = false)

    modell = RobustModels.rlm(f, daten, estimator(); σ0 = :mad, maxiter = 1000)
    if ReturnModell == true
        return modell
    else
        return StatsAPI.coef(modell)
    end
end

"""
 # mm_regression_fit(formula, daten::AbstractDataFrame; maxiter = 1000) -> RobustLinearModel

 Fit-Variante der robusten MM-Regression: gibt im Gegensatz zu `MM_Core` das
 ganze Modell zurück, damit darauf `StatsAPI.predict` angewendet werden kann
 (z.B. Cross-Validation in Regressionsvalidation.jl).

 ## Argumente

 - `maxiter`: Maximale Anzahl IRLS-Iterationen (S-Schritt und MM-Schritt).
   RobustModels defaultet ohne Angabe auf 30; hier wie in `MM_Core` Standard
   1000.
"""
function mm_regression_fit(formula, daten::AbstractDataFrame; maxiter::Integer = 1000)
    # HINWEIS PosDefException (RobustModels v0.6.0, beobachtet 2026-10):
    # Der S-Schritt des Tukey-MM-Schaetzers schlaegt auf "zu sauberen"
    # Trainingsdaten fehl (sigma schrumpft, alle Tukey-Gewichte -> 0,
    # gewichtete Cholesky singulaer). Tritt z.B. in LOO-Schleifen auf,
    # wenn das ausgelassene Subset keinen Ausreisser mehr enthaelt
    # (loo_mad_diagnostics/leave_one_out_cross_validation mit :MM).
    # Moegliche Reaktion, falls das auf echten Daten auftritt:
    # try/catch um den rlm-Aufruf und Rueckfall auf den konvergierenden
    # M-Estimator, also
    #   RobustModels.rlm(formula, daten,
    #       RobustModels.MEstimator{RobustModels.TukeyLoss}(); σ0 = :mad)
    return RobustModels.rlm(
        formula,
        daten,
        RobustModels.MMEstimator{RobustModels.TukeyLoss}();
        σ0 = :mad,
        maxiter = maxiter
    )
end