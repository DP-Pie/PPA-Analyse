using DataFrames
using GLM
using LinearAlgebra
using Random
using Statistics
using StatsAPI
using StatsModels

# =========================================================================
# Tools zum Validieren der Regressionen
# =========================================================================

"""
 # QDK(Q1::Float64, Median::Float64, Q3::Float64) -> Float64

 Berechnet den Quartilsdispersionskoeffizienten (QDK) aus bereits vorberechneten Quantilen.

 Der QDK ist ein robustes, dimensionsloses Streumaß und ist definiert als:

    QDK = (Q3 - Q1) / Median

 ## Argumente

 - `Q1::Float64`: 25 %-Quantil der Daten.
 - `Median::Float64`: 50 %-Quantil (Median) der Daten.
 - `Q3::Float64`: 75 %-Quantil der Daten.

 ## Rückgabe

 - `Float64`: Der Quartilsdispersionskoeffizient.
 - `NaN`, falls `Median == 0.0`. In diesem Fall wird bewusst keine Exception geworfen (siehe Kommentar im Code), damit nachfolgende
  Berechnungen nicht abbrechen.

 ## Fehler

 - `ArgumentError`, falls `Q1 > Q3`.
"""
function QDK(Q1::Float64,Median::Float64,Q3::Float64)
    if Q1 > Q3 then
        Q1, Q3 = Q3, Q1
    end

    # Macht in späteren berechnungen Probleme. 
    #if Median == 0.0 
    #    throw(DivisionByZero("Median der Daten ist Null. Berechnung nicht möglich."))
    #end
    
    # Statt Exception: NaN zurückgeben, falls Median = 0
    Median == 0.0 && return NaN

    result = (Q3 - Q1) / Median

    return result
end

"""
 # QDK(Q1::Float64, Q3::Float64) -> Float64

 Berechnet den Quartilsdispersionskoeffizienten in der Summenform

    QDK = (Q3 - Q1) / (Q1 + Q3)

 aus bereits vorberechneten Quantilen (ohne Median). Sind die Argumente vertauscht (`Q1 > Q3`), werden sie intern getauscht; das Ergebnis ist nach dem Tausch immer nichtnegativ.

 Gibt `NaN` zurück, falls `Q1 + Q3 == 0.0`.
"""
function QDK(Q1::Float64,Q3::Float64)
    # Division durch Null?
    Q13 = Q1 + Q3
    if Q13 == 0.0
        return NaN
    end
    
    if Q1 > Q3 then
        Q1, Q3 = Q3, Q1
    end

    result = (Q3 - Q1) / Q13

    return result
end

"""
 # QDK(Vec::Vector{Float64}, MedianForm::Bool=true) -> Float64

 Berechnet den Quartilsdispersionskoeffizienten (QDK) aus einem Datenvektor. Die Quantile werden intern über `quantile(Vec, [0.25, 0.5, 0.75])` bestimmt.

 Bei `MedianForm == true` gilt die Median-Form:

    QDK = (Q3 - Q1) / Median

 Bei `MedianForm == false` gilt die Summenform:

    QDK = (Q3 - Q1) / (Q3 + Q1)

 ## Argumente
 
 - `Vec::Vector{Float64}`: Datenvektor.
 - `MedianForm::Bool`: Wählt die Berechnungsvariante
  (`true` = Division durch den Median, `false` = Division durch die
  Quantilsumme `Q1 + Q3`).

 # Fehler
 
 - `DivisionByZero`, falls der relevante Nenner (Median bzw. Quantilsumme) Null ist.
"""
function QDK(Vec::Vector{Float64},MedianForm::Bool=true)
    Q1, Median, Q3 = quantile(Vec,[0.25,0.5,0.75])

    if MedianForm == true
        if Median == 0.0 
            throw(DivisionByZero("Median der Daten ist Null. Berechnung nicht möglich."))
        end
        result = (Q3 - Q1) / Median
    else
        ∑Q = Q3 + Q3
        if ∑Q == 0.0
            throw(DivisionByZero("Summer der Datenquantile 3 und 1 ist Null. Berechnung nicht möglich."))          
        end
    end
    return result
end

# Erstellt eine einfache Oberfläche für eindache Regressionmodel bis y = b0 + b1*x + b2*y + b3*x*y
# Rückgabe x-Vektor, y-Vektor, z-Matrix
function Make_Surface(Modell,lim_x::Vector{Float64},lim_y::Vector{Float64};steps::Int64 = 10, Dims::Int64)
    if length(lim_x) != 2 || length(lim_x) != 2
        throw(ArgumentError("Flächengrenzen in x und y dürfen nur 2 Elemente haben"))
    end

    Coefs = StatsAPI.coef(Modell)

    # Gitter erstellen
    x = range(lim_x[1], lim_x[2], length=steps)
    y = range(lim_y[1], lim_y[2], length=steps)

    # Oberfläche: z = f(x) für alle y (Extrusion entlang y-Achse)
    if Dims == 1 
        mySurface = [Coefs[1] + Coefs[2]*xi for xi in x, yi in y]
    elseif Dims == 2
        mySurface = [Coefs[1] + Coefs[2]*xi + Coefs[3]*yi for xi in x, yi in y]
    elseif Dims == 3
        mySurface = [Coefs[1] + Coefs[2]*xi + Coefs[3]*yi + Coefs[4]*xi*yi for xi in x, yi in y]
    else
        throw(ArgumentError("Dims: $(Dims) noch nicht implementiert"))
    end

    return x, y, mySurface
end

# =========================================================================
# Zentrale Registry und Formel-Katalog für Bootstrap und Cross-Validation
# =========================================================================
#
# Aufbau:
#   1. Adapter mit einheitlichen Verträgen je Verfahren:
#         core(formula, daten)    -> Vector{Float64}
#             Koeffizientenvektor, Intercept zuerst (wie bei StatsModels/StatsAPI).
#         fit(formula, daten)     -> modell
#         predict(modell, daten)  -> Vector{Float64}
#      Der Bootstrap nutzt `core`, die Cross-Validation (LOOCV, k-fold) nutzt
#      `fit` und `predict`.
#   2. Registry REG_CORES (Verfahren) und Formel-Katalog FORMELN (Kurven).
#   3. bootstrap(...), loo_koeffizienten(...), leave_one_out_cross_validation(...)
#      und k_fold_cross_validation(...): DataFrame-Methode (allgemein) +
#      Teil-Methode.
#
# Neues Regressionsverfahren -> Core-/Fit-/Predict-Funktionen mit obigen
#                              Verträgen schreiben und in REG_CORES
#                              registrieren.
# Neue Kurve                -> Formel in FORMELN aufnehmen. Nichtlineare
#                              Terme über vorab berechnete Spalten im
#                              DataFrame (z.B. :rMenge2), nicht pro Iteration.
#
# Beispiel:
#   boot = bootstrap(Teiledict["KK-SK343-75-AA-002"], :TheilSen)
#   boot = bootstrap(PTFE_DF, :MM, FORMELN[:ptfe]; B = 200)
#   std.(eachcol(boot))          # Bootstrap-Standardfehler je Koeffizient
#   loo  = leave_one_out_cross_validation(Teiledict["KK-SK343-75-AA-002"], :MM)
#   kf   = k_fold_cross_validation(Teiledict["KK-SK343-75-AA-002"], :TheilSen; k = 5)
# -------------------------------------------------------------------------

"""
 # _fit_theil_sen / _predict_theil_sen

 Fit-/Predict-Adapter der Theil-Sen-Regression für die Cross-Validation.
 `fit` erlaubt nur Formeln der Art `y ~ x` (Prüfung wie in `Theil_Sen_Coefs`)
 und liefert ein Named Tuple mit Intercept, Steigung und dem Namen der
 erklärenden Spalte; `predict` wendet die Koeffizienten auf Testdaten an.
"""
function _fit_theil_sen(formula::StatsModels.FormulaTerm, daten::AbstractDataFrame)
    koeffizienten = Theil_Sen_Coefs(formula, DataFrame(daten))
    xvar = only(collect(StatsModels.termvars(formula.rhs)))
    return (intercept = koeffizienten[1], slope = koeffizienten[2], xvar = xvar)
end

function _predict_theil_sen(modell, daten::AbstractDataFrame)
    return modell.slope .* Float64.(daten[!, modell.xvar]) .+ modell.intercept
end

"""
 # _fit_mm / _predict_mm

 Fit-/Predict-Adapter der robusten MM-Regression für die Cross-Validation.
 `fit` speichert Formel und Schema des Trainings-Fits im Modell, damit
 `predict` die Designmatrix der Testdaten mit dem Trainings-Schema aufbaut.
"""
function _fit_mm(formula::StatsModels.FormulaTerm, daten::AbstractDataFrame)
    model_frame = StatsModels.ModelFrame(
        formula,
        daten;
        model = RobustModels.RobustLinearModel
    )
    modell = mm_regression_fit(model_frame.f, daten)
    return (model = modell, formula = model_frame.f, schema = model_frame.schema)
end

function _predict_mm(modell, daten::AbstractDataFrame)
    model_frame = StatsModels.ModelFrame(
        modell.formula,
        modell.schema,
        daten,
        RobustModels.RobustLinearModel
    )
    designmatrix = StatsModels.modelmatrix(model_frame)
    designmatrix isa AbstractVector && (designmatrix = reshape(designmatrix, 1, :))
    return StatsAPI.predict(modell.model, designmatrix)
end

"""
 # REG_CORES / _core(Reg::Symbol) -> NamedTuple(:core, :fit, :predict)

 Registry der verfügbaren Regressions-Verfahren für `bootstrap` und die
 Cross-Validation. Ein neues Verfahren braucht die drei Funktionen mit den
 obigen Verträgen und einen Eintrag in `REG_CORES`.
"""
const REG_CORES = Dict{Symbol, NamedTuple{(:core, :fit, :predict), Tuple{Function, Function, Function}}}(
    :TheilSen => (core = Theil_Sen_Coefs, fit = _fit_theil_sen, predict = _predict_theil_sen),
    :MM       => (core = MM_Core, fit = _fit_mm, predict = _predict_mm),
)

function _core(Reg::Symbol)
    haskey(REG_CORES, Reg) || throw(ArgumentError(
        "Regressions-Typ $(Reg) nicht gefunden.\n" *
        "Verfügbar: " * join(sort(collect(keys(REG_CORES))), ", ")))
    return REG_CORES[Reg]
end

"""
 # FORMELN

 Formel-Katalog für `bootstrap` und die Cross-Validation. Die Spalten der Formel müssen im übergebenen
 DataFrame vorhanden sein (`:Länge` z.B. nur in PTFE_DF, nicht in teil.df).
 Nichtlineare Terme als vorab berechnete Spalte anlegen (siehe :quad).
"""
const FORMELN = Dict{Symbol, StatsModels.FormulaTerm}(
    :linear => @formula(tges ~ rMenge),
    :ptfe   => @formula(tges ~ rMenge + rMenge & Länge),
    # yj_tges = yeo_johnson(tges) vorab anlegen: λ̂ = yeo_johnson_auf_teil!(teil)
    :yeojohnson => @formula(yj_tges ~ rMenge),
    # :quad => @formula(tges ~ rMenge + rMenge2),   # rMenge2 = rMenge .^ 2 vorab anlegen
    :linearBDE => @formula(Zeit ~ rMenge),
)

"""
 # bootstrap(daten::DataFrame, Reg::Symbol, f::StatsModels.FormulaTerm; B=1_000, min_n=20) -> DataFrame

 Prüft die Stabilität der Regressionskoeffizienten per Bootstrap
 (Resampling mit Zurücklegen). Pro Iteration werden `nrow(daten)` Zeilen
 gezogen und mit der Core-Funktion `Reg` an die Formel `f` gefittet.
 Bei mehreren Prädiktoren werden alle Spalten gemeinsam gezogen, damit
 Wechselwirkungsterme (z.B. `rMenge & Länge`) erhalten bleiben.

 ## Rückgabe

 DataFrame mit `B` Zeilen und einer Spalte je Koeffizient (Spaltennamen aus
 der Formel, Intercept zuerst). Bootstrap-Standardfehler je Koeffizient:
 `std.(eachcol(ergebnis))`.

 Fehlgeschlagene Fits (z.B. PosDefException des MM-Schätzers oder ein
 Resample ohne unterschiedliche x-Werte) ergeben eine NaN-Zeile, statt den
 ganzen Lauf abbrechen zu lassen. NaN-Zeilen bei Auswertungen herausfiltern
 (completecases prüft nur auf missing, nicht auf NaN), z.B.:
 `boot[vec(all(isfinite, Matrix(boot), dims = 2)), :]`.

 ## Argumente

 - `Reg`: Verfahren, siehe `keys(REG_CORES)` (:TheilSen, :MM).
 - `f`: Regressionsformel, siehe `FORMELN`.
 - `B`: Anzahl Bootstrap-Iterationen.
 - `min_n`: Mindestanzahl Datenpunkte.
"""
function bootstrap(daten::DataFrame, Reg::Symbol, f::StatsModels.FormulaTerm;
    B::Int = 1_000, min_n::Int = 20)

    foo = _core(Reg).core

    n = nrow(daten)
    n < min_n && throw(ArgumentError(
        "Bootstrap: nur $(n) Datenpunkte, mindestens $(min_n) erforderlich."))

    # Referenz-Modelframe EINMAL aufbauen: Koeffizientennamen und -anzahl
    mf = StatsModels.ModelFrame(f, daten)
    namen = Symbol.(StatsModels.coefnames(mf))

    # Fehlgeschlagene Fits ergeben NaN-Zeilen statt Abbruch; vorinitialisiert
    # mit NaN, damit unbeschriebene Zeilen nicht als Zufallswerte landen.
    boot = fill(NaN, B, length(namen))
    Threads.@threads for i in 1:B
        rng = Xoshiro(i)
        idx = rand(rng, 1:n, n)
        try
            boot[i, :] .= foo(f, daten[idx, :])
        catch
        end
    end

    return DataFrame(boot, namen)
end

"""
 # bootstrap(teil::Teil, Reg::Symbol, f=FORMELN[:linear]; B=1_000, min_n=20) -> DataFrame

 Wie `bootstrap(daten::DataFrame, ...)`: wählt aus `teil.df` genau die Spalten,
 die die Formel `f` benötigt (`StatsModels.termvars`), und lässt fehlende Werte weg.
"""
function bootstrap(teil::Teil, Reg::Symbol, f::StatsModels.FormulaTerm = FORMELN[:linear];
    B::Int = 1_000, min_n::Int = 20)

    daten = dropmissing(select(teil.df, collect(StatsModels.termvars(f))))
    return bootstrap(daten, Reg, f; B = B, min_n = min_n)
end

"""
 # loo_koeffizienten(daten, Reg, f = FORMELN[:linear]; min_n = 3) -> Named Tuple
 # loo_koeffizienten(teil, Reg, f = FORMELN[:linear]; kwargs...) -> Named Tuple

 Leave-one-out über die Koeffizienten — das Gegenstück zu `bootstrap()`:
 Pro Beobachtung i wird das Verfahren `Reg` auf der Formel `f` ohne die
 i-te Zeile refittet (Core-Funktion aus `REG_CORES`, damit alle Verfahren
 und Formeln abgedeckt sind). Liefert die Koeffizientenverteilung der LOO-
 Refits und die Vorhersage des jeweils ausgelassenen Punkts.

 ## Rückgabe

 Named Tuple `(koeffizienten, vorhersagen, n_fehler)`:
 - `koeffizienten`: DataFrame mit `n` Zeilen und einer Spalte je Koeffizient
   (Spaltennamen wie bei `bootstrap`, Intercept zuerst). Fehlgeschlagene
   Fits ergeben NaN-Zeilen (gleiche Toleranz wie `bootstrap`); Filter via
   `vec(all(isfinite, Matrix(erg.koeffizienten), dims = 2))`.
 - `vorhersagen`: LOO-Vorhersage der ausgelassenen Punkte (linearer
   Prädiktor `xᵢ' β⁽⁻ⁱ⁾`, NaN bei fehlgeschlagenem Fit).
 - `n_fehler`: Anzahl fehlgeschlagener Refits.

 ## Beispiel
 ```julia
 erg = loo_koeffizienten(Teiledict["KK-SK343-75-AA-002"], :TheilSen)
 erg = loo_koeffizienten(PTFE_DF, :MM, FORMELN[:ptfe])
 ```
"""
function loo_koeffizienten(daten::AbstractDataFrame, Reg::Symbol,
        f::StatsModels.FormulaTerm = FORMELN[:linear]; min_n::Int = 3)

    foo = _core(Reg).core

    n = nrow(daten)
    n - 1 < min_n && throw(ArgumentError(
        "LOO: nur $(n) Datenpunkte, mindestens $(min_n + 1) erforderlich " *
        "(jeder Trainings-Fold braucht $(min_n) Punkte)."))

    # Referenz-Modelframe EINMAL aufbauen: Koeffizientennamen und Designmatrix
    mf = StatsModels.ModelFrame(f, daten)
    namen = Symbol.(StatsModels.coefnames(mf))
    X = StatsModels.modelmatrix(mf)

    loo = fill(NaN, n, size(X, 2))
    vorhersagen = fill(NaN, n)
    n_fehler = 0

    for i in 1:n
        train = daten[filter(!=(i), 1:n), :]
        try
            koef = Float64.(foo(f, train))
            loo[i, :] .= koef
            # Vorhersage des ausgelassenen Punkts als linearer Prädiktor
            vorhersagen[i] = dot(X[i, :], koef)
        catch
            n_fehler += 1
        end
    end

    return (koeffizienten = DataFrame(loo, namen),
        vorhersagen = vorhersagen,
        n_fehler = n_fehler)
end

function loo_koeffizienten(teil::Teil, Reg::Symbol,
        f::StatsModels.FormulaTerm = FORMELN[:linear]; kwargs...)
    daten = dropmissing(select(teil.df, collect(StatsModels.termvars(f))))
    return loo_koeffizienten(daten, Reg, f; kwargs...)
end

# =========================================================================
# Cross-Validation: Leave-one-out und k-fold
# =========================================================================
#
# Nutzt dieselbe Registry REG_CORES und denselben Formel-Katalog FORMELN wie
# der Bootstrap. Pro Fold werden Modell (`fit`) und Vorhersagen (`predict`)
# neu berechnet; die Kennzahlen werden über alle Folds aggregiert.
#
# Beispiel:
#   loo = leave_one_out_cross_validation(Teiledict["KK-SK343-75-AA-002"], :TheilSen)
#   kf  = k_fold_cross_validation(Teiledict["KK-SK343-75-AA-002"], :MM; k = 5)
# -------------------------------------------------------------------------

function _cross_validation_daten(daten::AbstractDataFrame, formula)
    variablen = unique(StatsModels.termvars(formula))
    isempty(variablen) && throw(ArgumentError(
        "Die Formel muss mindestens eine Datensatzvariable verwenden."
    ))

    fehlende_spalten = setdiff(variablen, propertynames(daten))
    isempty(fehlende_spalten) || throw(ArgumentError(
        "Formelspalten fehlen in den Daten: $(join(fehlende_spalten, ", "))."
    ))

    zeilen = findall(completecases(select(daten, variablen)))
    daten_vollständig = DataFrame(daten[zeilen, variablen])
    disallowmissing!(daten_vollständig)

    length(zeilen) >= 3 || throw(ArgumentError(
        "Für Cross-Validation werden mindestens drei vollständige Beobachtungen benötigt."
    ))

    beobachtet = Float64.(
        StatsModels.response(StatsModels.ModelFrame(formula, daten_vollständig))
    )
    all(isfinite, beobachtet) || throw(ArgumentError(
        "Die Zielvariable der Formel enthält nicht-endliche Werte."
    ))

    return zeilen, daten_vollständig, beobachtet
end

function _cross_validation_metrics(zeilen, beobachtet, vorhergesagt, fold_ids)
    all(isfinite, vorhergesagt) || throw(ArgumentError(
        "Die Regressionsfunktion hat nicht-endliche Vorhersagen erzeugt."
    ))

    residuen = beobachtet .- vorhergesagt
    quadratsumme = sum((beobachtet .- mean(beobachtet)) .^ 2)
    r2 = quadratsumme == 0 ? NaN : 1 - sum(residuen .^ 2) / quadratsumme

    vorhersagen_df = DataFrame(
        row = zeilen,
        fold = fold_ids,
        actual = beobachtet,
        predicted = vorhergesagt,
        residual = residuen
    )
    kennzahlen_df = DataFrame(
        n = length(zeilen),
        MAE = mean(abs.(residuen)),
        RMSE = sqrt(mean(residuen .^ 2)),
        R2 = r2
    )

    return (predictions = vorhersagen_df, metrics = kennzahlen_df)
end

function _cross_validate(
    regression,
    folds,
    zeilen,
    daten,
    beobachtet,
    formula
)
    vorhergesagt = Vector{Float64}(undef, length(zeilen))
    fold_ids = Vector{Int}(undef, length(zeilen))

    for (fold_id, test_indices) in enumerate(folds)
        isempty(test_indices) && throw(ArgumentError(
            "Fold $fold_id enthält keine Testbeobachtungen."
        ))
        trainingsmaske = trues(length(zeilen))
        trainingsmaske[test_indices] .= false
        trainingsindices = findall(trainingsmaske)
        isempty(trainingsindices) && throw(ArgumentError(
            "Fold $fold_id lässt keine Trainingsbeobachtungen übrig."
        ))

        trainingsdaten = daten[trainingsindices, :]
        testdaten = daten[test_indices, :]
        modell = regression.fit(formula, trainingsdaten)
        vorhersage = Float64.(regression.predict(modell, testdaten))
        length(vorhersage) == length(test_indices) || throw(ArgumentError(
            "Die Regressionsfunktion lieferte für Fold $fold_id " *
            "$(length(test_indices)) Testbeobachtungen $(length(vorhersage)) Vorhersagen."
        ))
        vorhergesagt[test_indices] = vorhersage
        fold_ids[test_indices] .= fold_id
    end

    return _cross_validation_metrics(zeilen, beobachtet, vorhergesagt, fold_ids)
end

"""
 # leave_one_out_cross_validation(daten, Reg, f=FORMELN[:linear])
 # leave_one_out_cross_validation(teil, Reg, f=FORMELN[:linear])

 Führt eine Leave-one-out-Kreuzvalidierung auf den vollständigen Fällen der
 Daten aus. Pro Beobachtung wird das Modell ohne diese Beobachtung gefittet
 (`fit`) und ihre Vorhersage (`predict`) berechnet.

 `Reg` ist ein Symbol aus `keys(REG_CORES)` (:TheilSen, :MM), `f` eine Formel
 aus `FORMELN`.

 ## Rückgabe

 Named Tuple `(predictions, metrics)`:
 - `predictions`: DataFrame mit `row` (ursprüngliche Zeile im übergebenen
   DataFrame bzw. `teil.df`), `fold`, `actual`, `predicted`, `residual`.
 - `metrics`: DataFrame mit `n`, `MAE`, `RMSE`, `R2`.

 ## Beispiel
 ```julia
 loo_ts = leave_one_out_cross_validation(teil, :TheilSen)
 loo_mm = leave_one_out_cross_validation(teil, :MM, FORMELN[:linear])
 ```
"""
function leave_one_out_cross_validation(
    daten::AbstractDataFrame,
    Reg::Symbol,
    formula::StatsModels.FormulaTerm = FORMELN[:linear]
)
    regression = _core(Reg)
    zeilen, daten_cv, beobachtet = _cross_validation_daten(daten, formula)
    folds = [[index] for index in 1:nrow(daten_cv)]
    return _cross_validate(regression, folds, zeilen, daten_cv, beobachtet, formula)
end

function leave_one_out_cross_validation(
    teil::Teil,
    Reg::Symbol,
    formula::StatsModels.FormulaTerm = FORMELN[:linear]
)
    return leave_one_out_cross_validation(teil.df, Reg, formula)
end

"""
 # k_fold_cross_validation(daten, Reg, f=FORMELN[:linear]; k=5, rng, shuffle=true)
 # k_fold_cross_validation(teil, Reg, f=FORMELN[:linear]; k=5, rng, shuffle=true)

 Führt eine k-fold-Kreuzvalidierung auf den vollständigen Fällen der Daten
 aus. Pro Fold werden Modell (`fit`) und Vorhersagen (`predict`) neu
 berechnet. `rng` steuert die reproduzierbare zufällige Aufteilung;
 `shuffle=false` verwendet die Eingabereihenfolge.

 `Reg` ist ein Symbol aus `keys(REG_CORES)` (:TheilSen, :MM), `f` eine Formel
 aus `FORMELN`.

 ## Rückgabe

 Wie `leave_one_out_cross_validation`: Named Tuple `(predictions, metrics)`;
 `predictions` enthält zusätzlich die Foldnummer je Beobachtung.

 ## Beispiel
 ```julia
 kf_mm = k_fold_cross_validation(teil, :MM; k = 5)
 ```
"""
function k_fold_cross_validation(
    daten::AbstractDataFrame,
    Reg::Symbol,
    formula::StatsModels.FormulaTerm = FORMELN[:linear];
    k::Integer = 5,
    rng::AbstractRNG = Random.default_rng(),
    shuffle::Bool = true
)
    regression = _core(Reg)
    zeilen, daten_cv, beobachtet = _cross_validation_daten(daten, formula)

    n = nrow(daten_cv)
    2 <= k <= n || throw(ArgumentError(
        "k muss zwischen 2 und der Anzahl vollständiger Beobachtungen ($n) liegen."
    ))

    indices = shuffle ? Random.randperm(rng, n) : collect(1:n)
    folds = [indices[fold_id:k:end] for fold_id in 1:k]
    return _cross_validate(regression, folds, zeilen, daten_cv, beobachtet, formula)
end

function k_fold_cross_validation(
    teil::Teil,
    Reg::Symbol,
    formula::StatsModels.FormulaTerm = FORMELN[:linear];
    k::Integer = 5,
    rng::AbstractRNG = Random.default_rng(),
    shuffle::Bool = true
)
    return k_fold_cross_validation(
        teil.df,
        Reg,
        formula;
        k = k,
        rng = rng,
        shuffle = shuffle
    )
end

function robuste_skala_mad(residuen)
    zentrum = median(residuen)
    return 1.4826 * median(abs.(residuen .- zentrum))
end

"""
 # loo_mad_diagnostics(df::DataFrame, formel; auswahl = :alle) -> DataFrame
 # loo_mad_diagnostics(daten::AbstractDataFrame, Reg::Symbol, formula = FORMELN[:linear]; auswahl = :alle) -> DataFrame
 # loo_mad_diagnostics(teil::Teil, ...; auswahl = :alle) -> DataFrame

 Leave-one-out-Diagnose (LOO) mit robuster  median-absolute-deviation-Skala (MAD). Pro Beobachtung i wird das
 Modell ohne i gefittet (`GLM.lm`) und daraus werden die Diagnosegrößen

 - `t_i = e_i^LOO / (s_i * sqrt(1 + q_i))` — robust standardisiertes LOO-Residuum,
 - `q_i = x_i' (X^(-i)' X^(-i))⁻¹ x_i` — LOO-Vorhersagehebel,
 - `D_i = (e_i^LOO)² q_i / (p s_i² (1 + q_i)) = (q_i / p) * t_i²` — robuste
   Cook's Distance,
 - `s_i` — robuste MAD-Skala der Trainingsresiduen,

 berechnet mit `e_i^LOO = y_i - ŷ_i^(-i)`, `s_i = 1.4826 * MAD` der
 Trainingsresiduen ohne i und `p` = Anzahl Modellkoeffizienten (inkl.
 Intercept).

 Referenzschwelle für `cooks_mad` wie beim klassischen Gegenstück heuristisch
 4/n; wegen der robusten Skala ist sie nur als grober Anhaltspunkt zu
 verstehen.

 ## Aufrufe

 - `loo_mad_diagnostics(df, formel; auswahl)` — klassische Regression mit
   `GLM.lm`.
 - `loo_mad_diagnostics(daten, Reg, formula; auswahl)` — Verfahren aus der
   Registry `REG_CORES` (`:TheilSen`, `:MM`) über deren `fit`/`predict`-
   Adapter. `formula` ist ein `FormulaTerm` oder ein Symbol aus dem
   Formel-Katalog `FORMELN` (`:linear`, `:ptfe`), Standard `FORMELN[:linear]`.
 - `loo_mad_diagnostics(teil, ...)` — wie oben, arbeitet auf `teil.df`.

 `hebel_loo` und `hii` sind unabhängig vom Verfahren reine Design-Größen
 (gewöhnliche Leverage der Designmatrix); `rstudent_mad`, `cooks_mad` und
 `skala_mad` nutzen die Residuen des jeweils gewählten Verfahrens.

 ## Argumente

 - `auswahl`: Symbol, Tuple/Vector von Symbolen oder `:alle` (Standard).
   Steuert, welche Diagnose-Spalten zusätzlich zu den Basis-Spalten
   (`row`, `actual`, `predicted`, `residual`) zurückgegeben werden:
     - `:rstudent` → `rstudent_mad` (t_i)
     - `:hebel`   → `hebel_loo` (q_i) und `hii` (klassische Hebelwert-Diagonale
       h_ii = q_i / (1 + q_i), wie in `ausreißer_flags` mit 2p/n-Schwelle genutzt)
     - `:cooks`    → `cooks_mad` (D_i)
     - `:skala`    → `skala_mad` (s_i)

 ## Rückgabe

 DataFrame mit den Basis-Spalten plus den gewählten Diagnose-Spalten.
 `rstudent_mad` und `cooks_mad` sind `missing`, wenn `s_i = 0` oder nicht
 endlich (MAD der Trainingsresiduen Null); `hebel_loo`, `hii` und `skala_mad`
 sind in dem Fall trotzdem definiert.

 ## Beispiel

 ```julia
 erg = loo_mad_diagnostics(teil.df, @formula(tges ~ rMenge))
 nur_cooks = loo_mad_diagnostics(teil.df, @formula(tges ~ rMenge); auswahl = :cooks)
 hebel_und_cooks = loo_mad_diagnostics(teil.df, @formula(tges ~ rMenge); auswahl = (:hebel, :cooks))
 erg = loo_mad_diagnostics(teil, :MM)
 erg = loo_mad_diagnostics(teil, :TheilSen, :linear; auswahl = (:hebel, :cooks))
 ```

 ## Fehler

 - `ArgumentError` bei unbekanntem Symbol in `auswahl` oder leerer Auswahl,
   unbekanntem Regressions-Typ `Reg` oder unbekanntem Formel-Symbol.
"""
function loo_mad_diagnostics(df::DataFrame, formel::StatsModels.FormulaTerm; auswahl = :alle)
    return _loo_mad_diagnostics(df, formel,
        (f, train) -> GLM.lm(f, train),
        (modell, ziel) -> StatsAPI.predict(modell, ziel);
        auswahl = auswahl)
end

function loo_mad_diagnostics(
    daten::AbstractDataFrame,
    Reg::Symbol,
    formula = FORMELN[:linear];
    auswahl = :alle
)
    # Hilfreicher Hinweis, wenn versehentlich ein Formel-Symbol als Verfahren übergeben wird
    if !haskey(REG_CORES, Reg) && haskey(FORMELN, Reg)
        throw(ArgumentError(
            ":$Reg ist eine Formel aus FORMELN, kein Regressions-Typ. " *
            "Gemeint ist wohl loo_mad_diagnostics(daten, :MM, $Reg) — " *
            "Verfügbar: " * join(sort(collect(keys(REG_CORES))), ", ")))
    end

    regression = _core(Reg)
    formel = _formel_aus_katalog(formula)
    return _loo_mad_diagnostics(daten, formel, regression.fit, regression.predict;
        auswahl = auswahl)
end

function loo_mad_diagnostics(teil::Teil, formel::StatsModels.FormulaTerm; auswahl = :alle)
    return loo_mad_diagnostics(teil.df, formel; auswahl = auswahl)
end

function loo_mad_diagnostics(
    teil::Teil,
    Reg::Symbol,
    formula = FORMELN[:linear];
    auswahl = :alle
)
    return loo_mad_diagnostics(teil.df, Reg, formula; auswahl = auswahl)
end

"""
 # _formel_aus_katalog(formel) -> StatsModels.FormulaTerm

 Akzeptiert einen `FormulaTerm` (unverändert) oder ein Symbol aus dem
 Formel-Katalog `FORMELN` und liefert die zugehörige Formel.
"""
function _formel_aus_katalog(formel)
    formel isa StatsModels.FormulaTerm && return formel
    formel isa Symbol || throw(ArgumentError(
        "Formel muss ein FormulaTerm oder ein Symbol aus FORMELN sein."))
    haskey(FORMELN, formel) || throw(ArgumentError(
        "Formel $(formel) nicht in FORMELN gefunden. " *
        "Verfügbar: " * join(sort(collect(keys(FORMELN))), ", ")))
    return FORMELN[formel]
end

"""
 # _loo_mad_diagnostics(df, formel, fit_fn, predict_fn; auswahl) -> DataFrame

 Kern der LOO-MAD-Diagnose mit austauschbaren Fit-/Predict-Adaptern
 (Verträge wie in REG_CORES: `fit_fn(formel, train_df)` bzw.
 `predict_fn(modell, daten_df)`). Öffentliche Aufrufe siehe
 `loo_mad_diagnostics`.
"""
function _loo_mad_diagnostics(
    df::AbstractDataFrame,
    formel::StatsModels.FormulaTerm,
    fit_fn::Function,
    predict_fn::Function;
    auswahl = :alle
)
    erlaubt = (:rstudent, :hebel, :cooks, :skala)

    # Auswahl normalisieren: einzelnes Symbol oder Sammlung von Symbolen
    symbole = auswahl isa Symbol ? (auswahl,) : Tuple(auswahl)
    if :alle in symbole || :all in symbole
        symbole = erlaubt
    end
    isempty(symbole) && throw(ArgumentError(
        "auswahl darf nicht leer sein. Verfügbar: " * join(erlaubt, ", ")))
    ungueltig = sort(collect(setdiff(Symbol.(symbole), erlaubt)))
    isempty(ungueltig) || throw(ArgumentError(
        "Unbekannte Auswahl: " * join(ungueltig, ", ") *
        ". Verfügbar: " * join(erlaubt, ", ") * " oder :alle"))
    symbole = Tuple(unique(Symbol.(symbole)))

    n = nrow(df)

    # Anzahl Modellkoeffizienten (inkl. Intercept) für die Cook-Skalierung
    p = size(StatsModels.modelmatrix(StatsModels.ModelFrame(formel, df)), 2)

    actual = Vector{Float64}(undef, n)
    predicted = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)
    hebel_loo = Vector{Float64}(undef, n)
    hii = Vector{Float64}(undef, n)
    skala_mad = Vector{Union{Missing, Float64}}(undef, n)
    rstudent_mad = Vector{Union{Missing, Float64}}(undef, n)
    cooks_mad = Vector{Union{Missing, Float64}}(undef, n)

    for i in 1:n
        train_indizes = filter(!=(i), 1:n)
        train = df[train_indizes, :]
        test = df[i:i, :]

        # Fit und Trainingsresiduen jeweils ohne Testkandidat i
        modell = fit_fn(formel, train)
        train_frame = StatsModels.ModelFrame(formel, train)
        test_frame = StatsModels.ModelFrame(
            train_frame.f,
            train_frame.schema,
            test,
            GLM.LinearModel
        )

        train_residuen =
            StatsModels.response(train_frame) .- predict_fn(modell, train)

        actual[i] = only(StatsModels.response(test_frame))
        predicted[i] = only(predict_fn(modell, test))
        residual[i] = actual[i] - predicted[i]

        # Designbasierter Vorhersagehebel des ausgelassenen Punkts
        X_train = Matrix(StatsModels.modelmatrix(train_frame))
        x_test = vec(StatsModels.modelmatrix(test_frame))
        q = dot(x_test, (X_train' * X_train) \ x_test) # Skalarprodukt qᵢ = xᵢᵀ(X₋ᵢᵀ X₋ᵢ)xᵢ

        skala = robuste_skala_mad(train_residuen)

        hebel_loo[i] = q
        hii[i] = q / (1 + q)
        skala_mad[i] = isfinite(skala) ? skala : missing

        # Bei MAD = 0 sind die standardisierten Größen nicht definiert.
        if isfinite(skala) && skala > 0
            t = residual[i] / (skala * sqrt(1 + q))
            rstudent_mad[i] = t
            # Robuste Cook's Distance: D_i = q_i / p * t_i²
            cooks_mad[i] = (q / p) * t^2
        else
            rstudent_mad[i] = missing
            cooks_mad[i] = missing
        end
    end

    # Basis-Spalten plus die gewählten Diagnose-Spalten
    spalten = Pair{Symbol, AbstractVector}[
        :row => collect(1:n),
        :actual => actual,
        :predicted => predicted,
        :residual => residual,
    ]
    :rstudent in symbole && push!(spalten, :rstudent_mad => rstudent_mad)
    if :hebel in symbole
        push!(spalten, :hebel_loo => hebel_loo)
        push!(spalten, :hii => hii)
    end
    :cooks in symbole && push!(spalten, :cooks_mad => cooks_mad)
    :skala in symbole && push!(spalten, :skala_mad => skala_mad)

    return DataFrame(spalten)
end

# =========================================================================
# Modellvergleich: Welcher Ansatz sagt die Wirklichkeit besser voraus?
# =========================================================================
#
# Prinzipien für einen fairen Vergleich:
#
#  1. Out-of-sample auf den ROH-DATEN (LOOCV / k-fold), nicht auf den bereits
#     ausreißerbereinigten Punkten — die Wirklichkeit enthält die Ausreißer,
#     und alle Modelle werden an denselben Beobachtungen gemessen.
#  2. Gemeinsame Skala: Vorhersagen transformierter Modelle werden mit dem
#     FOLD-EIGENEN λ̂ (nur aus den Trainingsdaten geschätzt, kein Leck)
#     zurück auf Sekunden transformiert, bevor Fehler berechnet werden.
#  3. Gepaarter Vergleich: alle Modelle werden auf denselben Zeilen bewertet;
#     schlägt ein Fit fehl (z.B. PosDefException beim MM-Schätzer), fällt die
#     Zeile für ALLE Modelle aus der Metrik, damit kein Modell bevorteilt wird.
#  4. Kennzahlen: MAE/MedAE (absolut), rel_MAE/rel_MedAE (relativ, fair bei
#     Heteroskedastizität), Kalibrierung (Anteil "real > vorhergesagt",
#     ~0.5 für ein Medianmodell) und RMSE nur zur Orientierung, da es von
#     Extremwerten dominiert wird.

"""
 # vergleiche_modelle(daten; modelle, k, rng, shuffle) -> Named Tuple
 # vergleiche_modelle(teil; ...) -> Named Tuple

 Kreuzvalidierter, gepaarter Modellvergleich der Regressionsansätze auf der
 Originalskala (Sekunden). Standardmäßig werden vier Modelle verglichen:

 - Theil-Sen linear        (`:TheilSen` auf `FORMELN[:linear]`)
 - MM-Tukey linear          (`:MM` auf `FORMELN[:linear]`)
 - Theil-Sen Yeo-Johnson    (`:TheilSen` auf `FORMELN[:yeojohnson]`, λ̂ je Fold)
 - MM-Tukey Yeo-Johnson     (`:MM` auf `FORMELN[:yeojohnson]`, λ̂ je Fold)

 ## Argumente

 - `modelle`: Vektor von NamedTuples mit `(name, Reg, yj)`; `yj = false` passt
   direkt auf `FORMELN[:linear]`, `yj = true` transformiert die Zielgröße im
   Fold (λ̂ nur aus Trainingsdaten).
 - `k`: `nothing` (Standard) = Leave-one-out; Ganzzahl = k-fold.
 - `rng`, `shuffle`: wie in `k_fold_cross_validation`.

 ## Rückgabe

 Named Tuple `(predictions, metrics, fit_fehler)`:
 - `predictions`: Long-DataFrame `row, Modell, actual, predicted, residual` —
   Grundlage für gepaarte Tests (z.B. Vorzeichen- oder Wilcoxon-Test auf den
   Residuen je Beobachtung).
 - `metrics`: eine Zeile je Modell mit `Modell, n, n_fit_fehler, MAE, RMSE,
   MedAE, rel_MAE, rel_MedAE, Kalibrierung` (auf den gemeinsamen Zeilen).
 - `fit_fehler`: Anzahl fehlgeschlagener Folds je Modell.

 ## Beispiel
 ```julia
 erg = vergleiche_modelle(Teiledict["KK-SK343-75-AA-002"])   # LOOCV
 erg.metrics
 ```
"""
function vergleiche_modelle(daten::AbstractDataFrame;
        modelle::AbstractVector{<:NamedTuple} = _standard_vergleich_modelle(),
        k::Union{Nothing, Integer} = nothing,
        rng::AbstractRNG = Random.default_rng(),
        shuffle::Bool = true)

    f_basis = FORMELN[:linear]   # rohe Zielgröße :tges als gemeinsamer Maßstab
    zeilen, daten_cv, beobachtet = _cross_validation_daten(daten, f_basis)

    n = nrow(daten_cv)
    if k === nothing
        folds = [[i] for i in 1:n]
    else
        2 <= k <= n || throw(ArgumentError(
            "k muss zwischen 2 und der Anzahl vollständiger Beobachtungen ($n) liegen."))
        indices = shuffle ? Random.randperm(rng, n) : collect(1:n)
        folds = [indices[fid:k:end] for fid in 1:k]
    end

    vorhersagen = Dict{String, Vector{Union{Missing, Float64}}}()
    fit_fehler = Dict{String, Int}()

    for model in modelle
        v = Vector{Union{Missing, Float64}}(missing, n)
        n_fehler = 0
        for test_idx in folds
            train_idx = setdiff(1:n, test_idx)
            train = daten_cv[train_idx, :]
            test = daten_cv[test_idx, :]
            try
                fit = _vergleich_fit(model, train)
                v[test_idx] .= _vergleich_predict(model, fit, test)
            catch
                n_fehler += 1
            end
        end
        vorhersagen[model.name] = v
        fit_fehler[model.name] = n_fehler
    end

    # Gemeinsame Zeilen: nur wo ALLE Modelle eine Vorhersage haben
    gemeinsam = findall(1:n) do i
        all(m -> !ismissing(vorhersagen[m.name][i]), modelle)
    end

    predictions = DataFrame()
    for model in modelle
        df = DataFrame(
            row = zeilen[gemeinsam],
            Modell = fill(model.name, length(gemeinsam)),
            actual = beobachtet[gemeinsam],
            predicted = Vector{Float64}(vorhersagen[model.name][gemeinsam])
        )
        df[!, :residual] = df.actual .- df.predicted
        predictions = vcat(predictions, df)
    end

    metrics = reduce(vcat, [
        _vergleich_kennzahlen(model.name,
            beobachtet[gemeinsam],
            Vector{Float64}(vorhersagen[model.name][gemeinsam]),
            fit_fehler[model.name])
        for model in modelle
    ])

    return (predictions = predictions, metrics = metrics, fit_fehler = fit_fehler)
end

function vergleiche_modelle(teil::Teil; kwargs...)
    return vergleiche_modelle(teil.df; kwargs...)
end

"""
 # vergleiche_modelle_alle_teile(Teiledict; min_n, k, ...) -> Named Tuple

 Führt `vergleiche_modelle` für jedes Teil mit mindestens `min_n` Einträgen
 aus und fasst die Ergebnisse zusammen. `k` sollte hier eine kleine Zahl
 (Standard 10) sein, da Leave-one-out über alle Teile teuer wird.

 ## Rückgabe

 Named Tuple `(pro_teil, vorhersagen, gesamt)`:
 - `pro_teil`: Metriken je Teil und Modell (Zeile = Teil × Modell).
 - `vorhersagen`: alle gepaarten Vorhersagen (Long-Format, mit `:Teil`).
 - `gesamt`: gepoolte Kennzahl je Modell. Relative Fehler und Kalibrierung
   sind über Teile hinweg interpretierbar; absolute MAE/RMSE sind es nicht,
   da die Zeitniveaus der Teile stark differieren.
"""
function vergleiche_modelle_alle_teile(teiledict::Dict{String, Teil};
        min_n::Int = 95,
        k::Integer = 10,
        modelle::AbstractVector{<:NamedTuple} = _standard_vergleich_modelle(),
        rng::AbstractRNG = Random.default_rng(),
        verbose::Bool = true)

    pro_teil = DataFrame[]
    vorhersagen = DataFrame[]

    for (name, teil) in teiledict
        teil.n >= min_n || continue
        try
            erg = vergleiche_modelle(teil; modelle = modelle, k = k, rng = rng)
            m = erg.metrics
            m[!, :Teil] .= name
            push!(pro_teil, m)
            p = erg.predictions
            p[!, :Teil] .= name
            push!(vorhersagen, p)
            verbose && println(name, ": OK (n = ", first(m.n), ")")
        catch e
            verbose && println(name, ": übersprungen (", typeof(e).name.name, ")")
        end
    end

    pro_teil_df = reduce(vcat, pro_teil)
    vorhersagen_df = reduce(vcat, vorhersagen)

    gesamt = combine(groupby(vorhersagen_df, :Modell)) do sdf
        rel_fehler = abs.(sdf.residual) ./ sdf.actual
        (rel_MAE_pooled = mean(rel_fehler),
         rel_MedAE_pooled = median(rel_fehler),
         Kalibrierung_pooled = mean(sdf.actual .> sdf.predicted),
         n_beobachtungen = nrow(sdf))
    end

    return (pro_teil = pro_teil_df, vorhersagen = vorhersagen_df, gesamt = gesamt)
end

# Standardmodell-Katalog für den Vergleich
function _standard_vergleich_modelle()
    return [
        (name = "Theil-Sen linear",  Reg = :TheilSen, yj = false),
        (name = "MM-Tukey linear",   Reg = :MM,       yj = false),
        (name = "Theil-Sen YJ",      Reg = :TheilSen, yj = true),
        (name = "MM-Tukey YJ",       Reg = :MM,       yj = true),
    ]
end

# Fit eines Vergleichsmodells auf den Trainings-Fold. Für yj-Modelle wird λ̂
# NUR aus den Trainingsdaten geschätzt (kein Leck aus den Testdaten).
function _vergleich_fit(model::NamedTuple, train::AbstractDataFrame)
    if !model.yj
        return _core(model.Reg).fit(FORMELN[:linear], train)
    end

    λ̂ = yeo_johnson_λ(Float64.(train[!, :tges]))
    train_t = DataFrame(train)
    train_t[!, :yj_tges] = yeo_johnson(Float64.(train_t[!, :tges]), λ̂)

    if model.Reg == :TheilSen
        return (modell = TheilSenModell(FORMELN[:yeojohnson], train_t), λ = λ̂)
    elseif model.Reg == :MM
        mf = StatsModels.ModelFrame(FORMELN[:yeojohnson], train_t;
            model = RobustModels.RobustLinearModel)
        return (modell = mm_regression_fit(mf.f, train_t),
            λ = λ̂, formula = mf.f, schema = mf.schema)
    end
    throw(ArgumentError("Unbekanntes Verfahren für den Modellvergleich: $(model.Reg)"))
end

# Vorhersage auf der ORIGINALSKALA (Sekunden): yj-Modelle werden mit dem
# fold-eigenen λ̂ zurücktransformiert.
function _vergleich_predict(model::NamedTuple, fit, test::AbstractDataFrame)
    if !model.yj
        return Float64.(_core(model.Reg).predict(fit, test))
    end

    z = if model.Reg == :TheilSen
        StatsAPI.predict(fit.modell, test)
    else
        mf = StatsModels.ModelFrame(fit.formula, fit.schema, test,
            RobustModels.RobustLinearModel)
        X = StatsModels.modelmatrix(mf)
        X isa AbstractVector && (X = reshape(X, 1, :))
        StatsAPI.predict(fit.modell, X)
    end
    return Float64.(yeo_johnson_inv(z, fit.λ))
end

# Kennzahlen auf der Originalskala für ein Modell
function _vergleich_kennzahlen(name::String, actual::Vector{Float64},
        predicted::Vector{Float64}, n_fit_fehler::Int)

    e = actual .- predicted
    return DataFrame(
        Modell = name,
        n = length(e),
        n_fit_fehler = n_fit_fehler,
        MAE = mean(abs.(e)),
        RMSE = sqrt(mean(e .^ 2)),
        MedAE = median(abs.(e)),
        rel_MAE = mean(abs.(e) ./ actual),
        rel_MedAE = median(abs.(e) ./ actual),
        Kalibrierung = mean(actual .> predicted)
    )
end

# =========================================================================
# Modell-Diagnostik: Bootstrap und Leave-one-out je Koeffizient
# =========================================================================
#
# Läuft über die Punkte des GEPEICHERTEN Modells (Modell-Skala; bei YJ-Modellen
# also auf der transformierten Skala). Misst damit die Stabilität DIESES Fits
# auf seinen eigenen Daten — die "Wirklichkeits"-Vorhersagegüte auf Rohdaten
# liefert stattdessen vergleiche_modelle (siehe oben).

"""
 # diagnostik!(modell::RegressionsModell; B = 1000) -> RegressionsModell
 # diagnostik!(teil::Teil, name = "Theil-Sen"; kwargs...) -> RegressionsModell

 Führt Bootstrap (B Resamples der Modellpunkte) und Leave-one-out
 (n Refits, je ohne einen Punkt) über die Punkte des Modells aus — jeweils
 über die zentralen Funktionen `bootstrap()` (reproduciblen Xoshiro(i)-
 Seeds) und `loo_koeffizienten()` — und SPEICHERT die Kennzahlen in
 `modell.Diagnostik` (überschreibt einen früheren Stand).

 Unterstützte Modelltypen in `modell.modell`: `TheilSenModell` und
 `RobustModels.RobustLinearModel` (MM; die Formel kommt aus dem Modell).
 Hinweis zu MM: Jeder Bootstrap-Resample und jeder LOO-Refit ist ein
 vollständiger rlm-Fit — `B` bei MM entsprechend wählen (z.B. 100–500).
 Fehlgeschlagene Fits (z.B. PosDefException auf "zu sauberen" Resamples)
 zählen als NaN und laufen in `n_fehler` ein, statt den Lauf abzubrechen.

 ## Aufbau des Diagnostik-DataFrames (Long-Format)

 Eine Zeile je (Verfahren, Koeffizient) plus eine Modellzeile:

 - `Verfahren`: "Bootstrap" oder "LOO"
 - `Koeffizient`: "(Intercept)", Prädiktorname oder "(Modell)"
 - Koeffizientenzeilen (Bootstrap und LOO):
   `Schätzung` (Punktschätzung des Fits), `n` (gültige Fits), `n_fehler`
   (fehlgeschlagene Fits), `Mittelwert`, `Median`, `SE` (Standardabweichung
   der Fit-Verteilung; beim Bootstrap der klassische Bootstrap-Standardfehler),
   `rel_SE` (SE / |Schätzung|, dimensionslos), `Q1`/`Q3` (Quartile), `QDK`
   (Quartilsdispersionskoeffizient, Summenform — Achtung: bei Koeffizienten
   mit Vorzeichenwechsel nicht interpretierbar), `Q2_5`/`Q97_5` (95 %-Bereich),
   `Bias` (Mittelwert − Schätzung), `Range` (max − min) und
   `Max_Abweichung` (größter Einfluss EINES Punktes, nur LOO aussagekräftig).
 - Modellzeile (Verfahren "LOO", Koeffizient "(Modell)") — Vorhersagegüte
   für den jeweils ausgelassenen Punkt: `MAE`, `RMSE`, `MedAE`, `rel_MedAE`,
   `R2`, `Kalibrierung` (Anteil "real > vorhergesagt", ~0.5 erwartet).

 Nicht zutreffende Spalten sind `missing`.

 ## Beispiel
 ```julia
 theil_sen_regression!(teil)
 diagnostik!(teil, "Theil-Sen"; B = 500)
 teil.RegressionsModell["Theil-Sen"].Diagnostik
 ```
"""
function diagnostik!(modell::RegressionsModell; B::Int = 1000)

    # Verfahren, Formel und Modell-Daten je Modelltyp rekonstruieren
    Reg, formel, daten = _diagnostik_daten(modell.modell, modell.points)
    schätzung = Float64.(StatsAPI.coef(modell.modell))

    # --- Bootstrap der Koeffizienten: zentrale bootstrap()-Funktion ---------
    # deckt alle Verfahren aus REG_CORES und alle Formeln ab; die Punkte des
    # Modells (Modell-Skala) werden unter ihren Original-Spaltennamen
    # rekonstruiert, damit die Formel des Fits unverändert passt. Fehl-
    # geschlagene Resamples stehen als NaN-Zeilen im Ergebnis.
    boot_df = bootstrap(daten, Reg, formel; B = B, min_n = 3)
    boot = Matrix(boot_df)
    koef_namen = string.(names(boot_df))     # Intercept zuerst, wie die Schätzung
    boot_ok = boot[vec(.!any(isnan.(boot), dims = 2)), :]
    b_fehler = B - size(boot_ok, 1)

    # --- Leave-one-out: zentrale loo_koeffizienten()-Funktion ---------------
    # Refittet je ohne einen Punkt über die Core-Funktion aus REG_CORES und
    # liefert Koeffizienten je Refit plus Vorhersage des ausgelassenen Punkts.
    loo_erg = loo_koeffizienten(daten, Reg, formel; min_n = 3)
    loo = Matrix(loo_erg.koeffizienten)
    loo_ok = loo[vec(.!any(isnan.(loo), dims = 2)), :]
    l_fehler = loo_erg.n_fehler
    loo_vorhersage = loo_erg.vorhersagen

    # --- Zusammenfassen ------------------------------------------------------
    zeilen = Vector{NamedTuple}()
    p = length(koef_namen)

    for k in 1:p
        isempty(boot_ok) || push!(zeilen,
            _diagnostik_zeile("Bootstrap", koef_namen[k], schätzung[k], boot_ok[:, k], b_fehler))
        isempty(loo_ok) || push!(zeilen,
            _diagnostik_zeile("LOO", koef_namen[k], schätzung[k], loo_ok[:, k], l_fehler))
    end

    # Modellzeile: LOO-Vorhersagegüte (Vorhersage des jeweils ausgelassenen Punkts)
    yvar = only(collect(StatsModels.termvars(formel.lhs)))
    y = Float64.(daten[!, yvar])
    ok = .!isnan.(loo_vorhersage)
    if any(ok)
        e = y[ok] .- loo_vorhersage[ok]
        sst = sum((y[ok] .- mean(y[ok])) .^ 2)
        r2 = sst == 0 ? NaN : 1 - sum(e .^ 2) / sst
        medae = median(abs.(e))
        push!(zeilen, (
            Verfahren = "LOO", Koeffizient = "(Modell)", Schätzung = missing,
            n = count(ok), n_fehler = l_fehler,
            Mittelwert = missing, Median = missing, SE = missing, rel_SE = missing,
            Q1 = missing, Q3 = missing, QDK = missing,
            Q2_5 = missing, Q97_5 = missing,
            Bias = missing, Range = missing, Max_Abweichung = missing,
            MAE = mean(abs.(e)), RMSE = sqrt(mean(e .^ 2)), MedAE = medae,
            rel_MedAE = medae / median(y[ok]),
            R2 = r2,
            Kalibrierung = mean(y[ok] .> loo_vorhersage[ok]),
        ))
    end

    modell.Diagnostik = DataFrame(zeilen)
    return modell
end

# Variante mit einer Abschätzung für die Iterationstiefe über die Messpunkteanzahl. 
function diagnostik!(modell::RegressionsModell)
    n = nrow(modell.points)
    if n < 40 
        B = 10_000
    elseif 40 <= n <= 100
        B = round(Int64, 29_240 * exp(-0.027 * n))
    elseif 100 < n <= 200
        B = round(Int64, 4_000 * exp(-0.007 * n))
    else 
        B = 1_000 
    end
    return diagnostik!(modell; B = B)
end

# Verfahren, Formel und Modell-Daten für diagnostik! aus dem gespeicherten
# Modell rekonstruieren. Theil-Sen kennt Formel und Spaltennamen direkt; für
# MM kommen die Formel aus dem RobustLinearModel und die Prädiktorspalten
# aus points (bei genau einem Prädiktor heißt die Spalte dort :x).
function _diagnostik_daten(ts::TheilSenModell, points::AbstractDataFrame)
    daten = DataFrame(
        ts.yvar => Float64.(points.y),
        ts.xvar => Float64.(points.x)
    )
    return (:TheilSen, ts.formel, daten)
end

function _diagnostik_daten(m::RobustModels.RobustLinearModel, points::AbstractDataFrame)
    RobustModels.hasformula(m) || throw(ArgumentError(
        "MM-Modell wurde ohne Formel gefittet; Diagnostik nicht möglich."))
    formel = StatsModels.formula(m)
    yvar = only(collect(StatsModels.termvars(formel.lhs)))
    xvars = collect(StatsModels.termvars(formel.rhs))

    daten = DataFrame(yvar => Float64.(points[!, :y]))
    for xv in xvars
        quelle = (length(xvars) == 1 && :x in propertynames(points)) ? :x : xv
        daten[!, xv] = Float64.(points[!, quelle])
    end
    return (:MM, formel, daten)
end

function _diagnostik_daten(m, points)
    throw(ArgumentError(
        "diagnostik! unterstützt modell.modell::TheilSenModell und " *
        "::RobustModels.RobustLinearModel, erhalten: $(typeof(m))."))
end

function diagnostik!(teil::Teil, name::String = "Theil-Sen"; kwargs...)
    haskey(teil.RegressionsModell, name) || throw(ArgumentError(
        "Kein Regressionsmodell '$(name)' für Teil $(teil.name). " *
        "Vorhanden: " * join(sort(collect(keys(teil.RegressionsModell))), ", ")))
    return diagnostik!(teil.RegressionsModell[name]; kwargs...)
end

# Eine Koeffizienten-Zeile des Diagnostik-DataFrames
function _diagnostik_zeile(verfahren::String, kname::String,
        schätzung::Float64, werte::AbstractVector{Float64}, n_fehler::Int)

    q = quantile(werte, [0.025, 0.25, 0.5, 0.75, 0.975])
    se = std(werte)
    return (
        Verfahren = verfahren,
        Koeffizient = kname,
        Schätzung = schätzung,
        n = length(werte),
        n_fehler = n_fehler,
        Mittelwert = mean(werte),
        Median = q[3],
        SE = se,
        rel_SE = schätzung == 0.0 ? missing : se / abs(schätzung),
        Q1 = q[2],
        Q3 = q[4],
        QDK = QDK(q[2], q[4]),
        Q2_5 = q[1],
        Q97_5 = q[5],
        Bias = mean(werte) - schätzung,
        Range = maximum(werte) - minimum(werte),
        Max_Abweichung = maximum(abs.(werte .- schätzung)),
        MAE = missing, RMSE = missing, MedAE = missing, rel_MedAE = missing,
        R2 = missing, Kalibrierung = missing,
    )
end


# =========================================================================
# Bewertung der Diagnostik: Stufe je Koeffizient
# =========================================================================
#
# Regelbasierte Einschätzung der Koeffizienten anhand der Diagnostik.
# Jedes Kriterium (Dimension) ergibt selbst eine Stufe; die finale Stufe
# ist das MINIMUM aller Dimensionen (Worst-Case-Kappung), und die
# Begründung nennt nur die Dimension(en), die tatsächlich bindend sind
# (Stufe = finale Stufe). Damit ist jede Stufe transparent begründbar —
# gleiche Philosophie wie das Ausreißer-Flag-System.
#
# Stufen (Wert → Label):
#   1 → "verwerfen"     2 → "untersuchen"     3 → "akzeptabel"
#   4 → "sicher"        5 → "sehr sicher"

const DIAGNOSTIK_STUFEN = ("verwerfen", "untersuchen", "akzeptabel", "sicher", "sehr sicher")

# Stufe aus einem Verhältnis, KLEIN = gut; schwellen = (sehr sicher, sicher,
# akzeptabel, untersuchen). Werte oberhalb von schwellen[4] → verwerfen.
function _stufe_aus_verhältnis(wert::Float64, schwellen)
    wert <= schwellen[1] && return 5
    wert <= schwellen[2] && return 4
    wert <= schwellen[3] && return 3
    wert <= schwellen[4] && return 2
    return 1
end

# Stufe aus einem Mindestwert, GROSS = gut; schwellen = (sehr sicher, sicher,
# akzeptabel, untersuchen). Werte unterhalb von schwellen[4] → verwerfen.
function _stufe_aus_mindestwert(wert::Float64, schwellen)
    wert >= schwellen[1] && return 5
    wert >= schwellen[2] && return 4
    wert >= schwellen[3] && return 3
    wert >= schwellen[4] && return 2
    return 1
end

"""
 # bewerte_diagnostik!(modell; kwargs...) -> RegressionsModell
 # bewerte_diagnostik!(teil, name = "Theil-Sen"; kwargs...) -> RegressionsModell

 Bewertet die Koeffizienten (und die Modellzeile) anhand der Diagnostik in
 `modell.Diagnostik` in fünf Stufen: sehr sicher, sicher, akzeptabel,
 untersuchen, verwerfen. Das Ergebnis wird als Spalten `Stufe_Wert`,
 `Stufe` und `Begründung` in den Diagnostik-DataFrame geschrieben (kein
 neues Struct-Feld); beide Zeilen eines Koeffizienten (Bootstrap und LOO)
 erhalten dieselbe Stufe, die Modellzeile "(Modell)" ihre eigene.

 Voraussetzung: `diagnostik!` wurde zuvor ausgeführt.

 ## Dimensionen je Koeffizient (Stufe = Minimum aller Dimensionen)

 - Rel. Bootstrap-Präzision `SE/|Schätzung|` — klein gut
   (`rel_se_schwellen`, Standard 0.05 / 0.10 / 0.25 / 0.50)
 - Vorzeichen-Sicherheit: enthält das 95 %-Bootstrap-Intervall die 0,
   wird auf "untersuchen" gekappt
 - Rel. LOO-Einzelpunkt-Einfluss `Max_Abweichung/|Schätzung|` — klein gut
   (`rel_loo_schwellen`, Standard 0.10 / 0.25 / 0.50 / 1.00)
 - Datenbasis `n_OK/Anzahl Koeffizienten` — groß gut
   (`punkte_je_koeffizient`, Standard 30 / 20 / 10 / 5)
 - Ausreißeranteil `pro_Ausreißer` aus stats — klein gut
   (`ausreißer_schwellen`, Standard 0.05 / 0.15 / 0.30 / 0.50; nur wenn
   stats die Spalte bereithält — aktuell Theil-Sen, MM-Long-Format ohne)
 - Anteil fehlgeschlagener Fits (`max_fit_fehler_anteil`, Standard 0.10)
   — Kappung auf "untersuchen"

 ## Dimensionen der Modellzeile "(Modell)"

 - Rel. Medianfehler `rel_MedAE` — klein gut
   (`rel_medae_schwellen`, Standard 0.10 / 0.20 / 0.35 / 0.50)
 - Kalibrierung: |Kalibrierung − 0,5| ≤ 0,05 → keine Kappung,
   ≤ 0,10 → akzeptabel, sonst "untersuchen" (ein Medianmodell sollte bei
   0,5 liegen)
 - Datenbasis, Fit-Fehler und Ausreißeranteil wie oben

 ## Beispiel
 ```julia
 diagnostik!(teil, "Theil-Sen"; B = 500)
 bewerte_diagnostik!(teil, "Theil-Sen")
 # Uebersicht je Koeffizient:
 combine(groupby(teil.RegressionsModell["Theil-Sen"].Diagnostik, :Koeffizient),
     :Stufe => first => :Stufe, :Begründung => first => :Begründung)
 ```
"""
function bewerte_diagnostik!(modell::RegressionsModell;
        rel_se_schwellen = (0.05, 0.10, 0.25, 0.50),
        rel_loo_schwellen = (0.10, 0.25, 0.50, 1.00),
        punkte_je_koeffizient = (30.0, 20.0, 10.0, 5.0),
        ausreißer_schwellen = (0.05, 0.15, 0.30, 0.50),
        rel_medae_schwellen = (0.10, 0.20, 0.35, 0.50),
        max_fit_fehler_anteil = 0.10)

    d = modell.Diagnostik
    isempty(d) && throw(ArgumentError(
        "Diagnostik ist leer — zuerst diagnostik!(modell; B = ...) ausführen."))

    n_ok = nrow(modell.points)
    koef_namen = unique(d.Koeffizient)
    p = count(!=("(Modell)"), koef_namen)

    # Ausreißeranteil nur bewerten, wenn stats ihn bereithält (Theil-Sen ja,
    # MM-Long-Format bisher ohne die Spalte → Dimension entfällt)
    hat_anteil = :pro_Ausreißer in propertynames(modell.stats)
    ausreißer_anteil = hat_anteil ? Float64(first(modell.stats.pro_Ausreißer)) : missing

    stufen = Dict{String, Int}()
    begründungen = Dict{String, String}()

    for kname in koef_namen
        zeilen_k = d.Koeffizient .== kname
        boot = d[zeilen_k .&& (d.Verfahren .== "Bootstrap"), :]
        loo = d[zeilen_k .&& (d.Verfahren .== "LOO"), :]

        # (Stufe, Begründungstext) je Dimension; Begründungstext nur gefüllt
        # bei Stufe < 5, damit nicht-bindende Dimensionen still bleiben.
        dimensionen = Tuple{Int, String}[]

        ist_modellzeile = kname == "(Modell)"
        schätzung = missing
        for r in eachrow(d[zeilen_k, :])
            if !ismissing(r.Schätzung)
                schätzung = Float64(r.Schätzung)
                break
            end
        end

        if ist_modellzeile
            # --- Modellzeile: Vorhersagegüte + Kalibrierung ---------------
            if nrow(loo) == 1 && !ismissing(loo.rel_MedAE[1])
                s = _stufe_aus_verhältnis(loo.rel_MedAE[1], rel_medae_schwellen)
                push!(dimensionen, (s, s < 5 ?
                    "relativer Medianfehler $(round(loo.rel_MedAE[1], digits = 2))" : ""))

                kal = loo.Kalibrierung[1]
                if abs(kal - 0.5) > 0.05
                    s = abs(kal - 0.5) <= 0.10 ? 3 : 2
                    push!(dimensionen, (s,
                        "Kalibrierung $(round(kal, digits = 2)) statt ~0,5"))
                end
            end
        else
            # --- Koeffizient: Präzision, Vorzeichen, Einzelpunkt-Einfluss --
            if ismissing(schätzung) || schätzung == 0.0
                push!(dimensionen, (2, "Schätzung ≈ 0, relative Maße nicht definiert"))
            else
                if nrow(boot) == 1 && !ismissing(boot.SE[1])
                    rel_se = boot.SE[1] / abs(schätzung)
                    s = _stufe_aus_verhältnis(rel_se, rel_se_schwellen)
                    push!(dimensionen, (s, s < 5 ?
                        "relativer Bootstrap-SE $(round(rel_se, digits = 2))" : ""))

                    if boot.Q2_5[1] * boot.Q97_5[1] <= 0
                        push!(dimensionen, (2, "95 %-Intervall enthält die 0"))
                    end
                elseif nrow(boot) == 0
                    push!(dimensionen, (2, "keine Bootstrap-Verteilung"))
                end

                if nrow(loo) == 1 && !ismissing(loo.Max_Abweichung[1])
                    rel_loo = loo.Max_Abweichung[1] / abs(schätzung)
                    s = _stufe_aus_verhältnis(rel_loo, rel_loo_schwellen)
                    push!(dimensionen, (s, s < 5 ?
                        "Einzelpunkt-Einfluss $(round(rel_loo, digits = 2))" : ""))
                end
            end

            if p > 0
                s = _stufe_aus_mindestwert(n_ok / p, punkte_je_koeffizient)
                push!(dimensionen, (s, s < 5 ?
                    "nur $(round(n_ok / p, digits = 1)) Punkte je Koeffizient" : ""))
            end
        end

        for zeile in (boot, loo)
            nrow(zeile) == 1 || continue
            gesamt = zeile.n[1] + zeile.n_fehler[1]
            gesamt > 0 || continue
            anteil = zeile.n_fehler[1] / gesamt
            if anteil > max_fit_fehler_anteil
                push!(dimensionen, (2,
                    "$(round(anteil * 100, digits = 1)) % fehlgeschlagene Fits"))
            end
        end

        if !ismissing(ausreißer_anteil)
            s = _stufe_aus_verhältnis(ausreißer_anteil, ausreißer_schwellen)
            push!(dimensionen, (s, s < 5 ?
                "Ausreißeranteil $(round(ausreißer_anteil * 100, digits = 1)) %" : ""))
        end

        # Finale Stufe = Minimum; Begründung = nur die bindenden Dimensionen
        stufe = isempty(dimensionen) ? 5 : minimum(s for (s, _) in dimensionen)
        stufen[kname] = stufe
        begründungen[kname] = join(
            [g for (s, g) in dimensionen if s == stufe && !isempty(g)], "; ")
    end

    d[!, :Stufe_Wert] = [stufen[k] for k in d.Koeffizient]
    d[!, :Stufe] = [DIAGNOSTIK_STUFEN[w] for w in d.Stufe_Wert]
    d[!, :Begründung] = [begründungen[k] for k in d.Koeffizient]

    return modell
end

function bewerte_diagnostik!(teil::Teil, name::String = "Theil-Sen"; kwargs...)
    haskey(teil.RegressionsModell, name) || throw(ArgumentError(
        "Kein Regressionsmodell '$(name)' für Teil $(teil.name). " *
        "Vorhanden: " * join(sort(collect(keys(teil.RegressionsModell))), ", ")))
    return bewerte_diagnostik!(teil.RegressionsModell[name]; kwargs...)
end
