using Random

function mm_regression_fit(formula, daten::AbstractDataFrame)
    return RobustModels.rlm(
        formula,
        daten,
        RobustModels.MMEstimator{RobustModels.TukeyLoss}();
        σ0 = :mad
    )
end

function _loo_model_frame(formula, daten::AbstractDataFrame)
    return StatsModels.ModelFrame(formula, daten)
end

function _loo_fit(::typeof(Theil_Sen_Core), formula, daten::AbstractDataFrame)
    model_frame = _loo_model_frame(formula, daten)
    x = StatsModels.modelmatrix(model_frame)

    ndims(x) == 2 && size(x, 2) == 2 || throw(ArgumentError(
        "Theil-Sen unterstützt nur eine erklärende Variable mit Intercept."
    ))

    y = StatsModels.response(model_frame)
    slope, intercept, _ = Theil_Sen_Core(Float64.(x[:, 2]), Float64.(y))
    return (slope = slope, intercept = intercept)
end

function _loo_fit(::typeof(mm_regression_fit), formula, daten::AbstractDataFrame)
    model_frame = StatsModels.ModelFrame(
        formula,
        daten;
        model = RobustModels.RobustLinearModel
    )
    modell = mm_regression_fit(model_frame.f, daten)
    return (model = modell, formula = model_frame.f, schema = model_frame.schema)
end

function _loo_fit(regression_function, formula, daten::AbstractDataFrame)
    return regression_function(formula, daten)
end

function _loo_predict(
    ::typeof(Theil_Sen_Core),
    modell,
    formula,
    daten::AbstractDataFrame
)
    model_frame = _loo_model_frame(formula, daten)
    x = StatsModels.modelmatrix(model_frame)
    ndims(x) == 2 && size(x, 2) == 2 || throw(ArgumentError(
        "Theil-Sen unterstützt nur eine erklärende Variable mit Intercept."
    ))
    return modell.slope .* Float64.(x[:, 2]) .+ modell.intercept
end

function _loo_predict(
    ::typeof(mm_regression_fit),
    modell,
    formula,
    daten::AbstractDataFrame
)
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

function _loo_predict(regression_function, modell, formula, daten::AbstractDataFrame)
    return StatsAPI.predict(modell, daten)
end

const LOOCV_REGRESSIONEN = Dict{Symbol, Function}(
    :theil_sen => Theil_Sen_Core,
    :mm => mm_regression_fit,
    :linear => GLM.lm
)

function _resolve_loocv_regression(name::Symbol)
    haskey(LOOCV_REGRESSIONEN, name) || throw(ArgumentError(
        "Unbekannte Regression :$name. Verfügbar sind: " *
        join(string.(sort!(collect(keys(LOOCV_REGRESSIONEN)))), ", ")
    ))
    return LOOCV_REGRESSIONEN[name]
end

function _cross_validation_data(teil::Teil, formula)
    variablen = unique(StatsModels.termvars(formula))
    isempty(variablen) && throw(ArgumentError(
        "Die Formel muss mindestens eine Datensatzvariable verwenden."
    ))

    fehlende_spalten = setdiff(variablen, propertynames(teil.df))
    isempty(fehlende_spalten) || throw(ArgumentError(
        "Formelspalten fehlen in Teil.df: $(join(fehlende_spalten, ", "))."
    ))

    relevante_daten = select(teil.df, variablen)
    zeilen = findall(completecases(relevante_daten))
    daten = DataFrame(teil.df[zeilen, variablen])
    disallowmissing!(daten)

    length(zeilen) >= 3 || throw(ArgumentError(
        "Für Cross-Validation werden mindestens drei vollständige Beobachtungen benötigt."
    ))

    beobachtet = Float64.(StatsModels.response(_loo_model_frame(formula, daten)))
    all(isfinite, beobachtet) || throw(ArgumentError(
        "Die Zielvariable der Formel enthält nicht-endliche Werte."
    ))

    return zeilen, daten, beobachtet
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
    regression_function,
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
        modell = _loo_fit(regression_function, formula, trainingsdaten)
        vorhersage = Float64.(_loo_predict(regression_function, modell, formula, testdaten))
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
 #  leave_one_out_cross_validation(teil, regression_function;
                                   formula = @formula(tges ~ rMenge))

 Führt eine Leave-one-out-Kreuzvalidierung auf den vollständigen Fällen in
 `teil.df` aus. `regression_function` muss entweder `Theil_Sen_Core` oder eine
 Fit-Funktion sein, die eine Formel und einen Trainings-DataFrame entgegennimmt;
 das resultierende Modell muss `StatsAPI.predict` unterstützen.

 Beispiele:
 ```julia
 loo_ts = leave_one_out_cross_validation(teil, Theil_Sen_Core)
 loo_mm = leave_one_out_cross_validation(teil, mm_regression_fit)
 loo_mm = leave_one_out_cross_validation(teil, :mm)
 loo_lm = leave_one_out_cross_validation(
     teil,
     :linear;
     formula = @formula(tges ~ rMenge + IstteSek)
 )
 ```

 Gibt ein Named Tuple mit den Vorhersagen je Beobachtung und den aggregierten
 Kennzahlen MAE, RMSE und R² zurück. Die Spalte `row` verweist auf die
 ursprüngliche Zeile in `teil.df`.
"""
function leave_one_out_cross_validation(
    teil::Teil,
    regression_function;
    formula = StatsModels.@formula(tges ~ rMenge)
)
    zeilen, daten, beobachtet = _cross_validation_data(teil, formula)
    folds = [[index] for index in 1:nrow(daten)]
    return _cross_validate(regression_function, folds, zeilen, daten, beobachtet, formula)
end

function leave_one_out_cross_validation(
    teil::Teil,
    regression_name::Symbol;
    formula = StatsModels.@formula(tges ~ rMenge)
)
    return leave_one_out_cross_validation(
        teil,
        _resolve_loocv_regression(regression_name);
        formula
    )
end

"""
 # k_fold_cross_validation(teil, regression_function; k=5, formula, rng, shuffle=true)

 Führt eine k-fold-Kreuzvalidierung auf den vollständigen Fällen in `teil.df`
 aus. Pro Fold werden Modell und Vorhersagen neu berechnet. `rng` steuert die
 reproduzierbare zufällige Aufteilung; `shuffle=false` verwendet die
 Eingabereihenfolge. Das Ergebnis enthält Vorhersagen je Beobachtung mit
 Foldnummer sowie aggregierte MAE-, RMSE- und R²-Kennzahlen.

 `regression_function` kann eine passende Fit-Funktion oder ein Symbol aus
 `LOOCV_REGRESSIONEN` sein (`:theil_sen`, `:mm`, `:linear`).
"""
function k_fold_cross_validation(
    teil::Teil,
    regression_function;
    k::Integer = 5,
    formula = StatsModels.@formula(tges ~ rMenge),
    rng::AbstractRNG = Random.default_rng(),
    shuffle::Bool = true
)
    zeilen, daten, beobachtet = _cross_validation_data(teil, formula)
    n = nrow(daten)
    2 <= k <= n || throw(ArgumentError(
        "k muss zwischen 2 und der Anzahl vollständiger Beobachtungen ($n) liegen."
    ))

    indices = shuffle ? Random.randperm(rng, n) : collect(1:n)
    folds = [indices[fold_id:k:end] for fold_id in 1:k]
    return _cross_validate(regression_function, folds, zeilen, daten, beobachtet, formula)
end

function k_fold_cross_validation(
    teil::Teil,
    regression_name::Symbol;
    k::Integer = 5,
    formula = StatsModels.@formula(tges ~ rMenge),
    rng::AbstractRNG = Random.default_rng(),
    shuffle::Bool = true
)
    return k_fold_cross_validation(
        teil,
        _resolve_loocv_regression(regression_name);
        k,
        formula,
        rng,
        shuffle
    )
end

using DataFrames
using GLM
using LinearAlgebra
using Statistics
using StatsAPI
using StatsModels

function robuste_skala_mad(residuen)
    zentrum = median(residuen)
    return 1.4826 * median(abs.(residuen .- zentrum))
end

function loo_robustes_rstudent(df::DataFrame, formel)
    n = nrow(df)

    actual = Vector{Float64}(undef, n)
    predicted = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)
    rstudent_mad = Vector{Union{Missing, Float64}}(undef, n)

    for i in 1:n
        train_indizes = filter(!=(i), 1:n)
        train = df[train_indizes, :]
        test = df[i:i, :]

        # Fit und Trainingsresiduen jeweils ohne Testkandidat i
        modell = GLM.lm(formel, train)
        train_frame = StatsModels.ModelFrame(formel, train)
        test_frame = StatsModels.ModelFrame(
            train_frame.f,
            train_frame.schema,
            test,
            GLM.LinearModel
        )

        train_residuen =
            StatsModels.response(train_frame) .- StatsAPI.predict(modell, train)

        actual[i] = only(StatsModels.response(test_frame))
        predicted[i] = only(StatsAPI.predict(modell, test))
        residual[i] = actual[i] - predicted[i]

        # Designbasierter Vorhersagehebel des ausgelassenen Punkts
        X_train = Matrix(StatsModels.modelmatrix(train_frame))
        x_test = vec(StatsModels.modelmatrix(test_frame))
        q = dot(x_test, (X_train' * X_train) \ x_test)

        skala = robuste_skala_mad(train_residuen)

        # Bei MAD = 0 ist das standardisierte Residuum nicht definiert.
        if isfinite(skala) && skala > 0
            rstudent_mad[i] =
                residual[i] / (skala * sqrt(1 + q))
        else
            rstudent_mad[i] = missing
        end
    end

    return DataFrame(
        row = 1:n,
        actual = actual,
        predicted = predicted,
        residual = residual,
        rstudent_mad = rstudent_mad
    )
end
