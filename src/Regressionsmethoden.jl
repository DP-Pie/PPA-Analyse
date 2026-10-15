# =========================================================================
# Datenvorbereitung
# =========================================================================
function PrepareData!(teil::Teil, Reg::Symbol = :MM, formel = FORMELN[:linear];
        t_rstudent::Float64 = 2.5,
        rstudent_faktor::Float64 = 3.0,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    ausreißer_flags_setzen!(teil, Reg, formel;
    t_rstudent,
    rstudent_faktor,
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
        rstudent_faktor::Float64 = 3.0,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    OK_df, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(teil, :TheilSen, formel;
        t_rstudent,
        rstudent_faktor,
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

 ## Quelle der Zeitgröße

 `quelle` (Standard `:tges`): Spalte der Zeitgröße in `teil.df`. Bei
 `quelle != :tges` wird die Quelle vorab als kanonische Spalte `:tges`
 in `teil.df` übernommen (eine vorhandene :tges-Spalte wird überschrieben),
 denn die Ausreißer-Diagnostik (UKZ-Regel), `FORMELN` und die gesamte
 nachgelagerte Pipeline (diagnostik!, Diagramme) arbeiten auf `:tges`.
 Beispiel BDE02: `theil_sen_regression_yj!(teil; quelle = :Zeit)`.

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
        quelle::Symbol = :tges,
        t_rstudent::Float64 = 2.5,
        rstudent_faktor::Float64 = 3.0,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2)

    # Kanonische :tges-Spalte bereitstellen; quelle hat Vorrang:
    # UKZ-Regel, FORMELN und diagnostik!/Diagramme arbeiten auf :tges
    quelle != :tges && (teil.df.tges = teil.df[!, quelle])

    # --- Pass 1: λ̂ auf Rohdaten, Flags auf der YJ-Skala -------------------
    yeo_johnson_auf_teil!(teil; quelle = quelle)      # λ̂-Pass 1, Spalte :yj_tges
    _, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(
        teil, :TheilSen, FORMELN[:yeojohnson];
        t_rstudent = t_rstudent,
        rstudent_faktor = rstudent_faktor,
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
    yeo_johnson_auf_teil!(teil; quelle = quelle, λ = λ̂)               # Spalte mit finalem λ̂
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
        rstudent_faktor::Float64 = 3.0,
        hebel_faktor::Float64 = 3.0,
        t_cooks::Float64 = 4.0,
        ukz_n::Float64 = 1.0,
        ukz_tges::Float64 = 200.0,
        verdacht_min::Int64 = 2,
        maxiter::Integer = 1000)

    OK_df, Messpunkte, Ausreißer, Ausreißer_pro = PrepareData!(teil, :MM, formel;
        t_rstudent,
        rstudent_faktor,
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

# =========================================================================
# Mixed-Model-Regression (MixedModels.jl)
# =========================================================================

"""
 # lmm_regression!(teile::Dict{String,Teil}; kwargs...) -> NamedTuple

 Lineares gemischtes Modell über ALLE Teile des Dictionaries mit partiellem
 Pooling der Gruppenkoeffizienten:

     tges ~ 1 + rMenge + (1 + rMenge | Gruppe)

 VORAUSSETZUNG: MixedModels muss geladen sein (`using MixedModels` in
 main.jl), da die Aufrufe von `fit(MixedModel, ...)` und `raneftables`
 erst zur Laufzeit aufgelöst werden. Notebook-Umgebungen ohne MixedModels
 können die Datei weiterhin includen — nur der Aufruf braucht das Paket.

 Anders als `theil_sen_regression_yj!`/`mm_regression!` (ein Aufruf = ein Teil)
 ist der Fit global: alle Messpunkte aller Teile fließen gemeinsam in EIN
 Modell ein. Ein Aufruf bearbeitet daher das ganze Dictionary und legt JEDEM
 Teil sein Gruppen-Ergebnis als `RegressionsModell "MixedModel"` in
 `teil.RegressionsModell` ab.

 - Fixe Effekte  (`c_Intercept`, `c_Slope`): globale Gerade über alle Gruppen.
 - Zufallseffekte (`rng_Intercept`, `rng_Slope`): Abweichung der Gruppe vom
   globalen Muster — Gruppen mit wenig Daten werden Richtung Pool geschrumpft.
 - Gruppengerade (`all_Intercept`, `all_Slope`): fix + zufällig; liegt als
   `stats.m` (Steigung) und `stats.n` (Achsenabschnitt) vor, wie bei
   Theil-Sen/MM, damit Auswertungen (ausschluss_grund, Meta-Regression)
   unverändert weiterlaufen.

 ## Argumente

 - `quelle`: Zeitspalte in `teil.df` (Standard `:tges`; AOI/BDE02: `:Zeit`).
 - `prädiktor`: Mengenspalte (Standard `:rMenge`).
 - `ausnehmen`: Teilnamen, die NICHT in den Fit eingehen (Standard
   `["Sonstiges"]` — gemischte Restgruppe, eine Regression darüber wäre
   bedeutungslos). Diese Teile erhalten auch kein `"MixedModel"`-Eintrag.
 - `ausreißer_filtern`: falls true (Standard), werden Zeilen mit Stufe-1-Flag
   `"Ausreißer"` vor dem Fit entfernt (analog zu AOI_fit_df beim Bayes-Weg);
   `"Verdacht"` bleibt drin. Voraussetzung: das Flag existiert (Spalte
   `:Ausreißer`, sonst fließen die Punkte ungefiltert ein.

 ## Rückgabe

 NamedTuple `(lmm, fix, gruppen)`:
 - `lmm`     ... das MixedModels-Objekt (StatsAPI.predict/-coef nutzbar)
 - `fix`     ... `DataFrame(coeftable(lmm))` der fixen Effekte
 - `gruppen` ... Übersicht je Gruppe: `c_/rng_/all_Intercept`, `c_/rng_/all_Slope`

 ## Beispiel
 ```julia
 erg = lmm_regression!(AOI_teile; quelle = :Zeit)
 vscodedisplay(erg.fix)
 vscodedisplay(erg.gruppen)
 liste_regressionsmodelle(AOI_teile["VAG01e"])   # enthält jetzt "MixedModel"
 ```
"""
function lmm_regression!(teile::Dict{String,Teil};
        quelle::Symbol = :tges,
        prädiktor::Symbol = :rMenge,
        ausnehmen::Vector{String} = String["Sonstiges"],
        ausreißer_filtern::Bool = true)

    # --- 1. gepoolten Datensatz über alle verwendeten Teile bauen ----------
    # je Teil: Ausreißer entfernen, auf die Regressions­spalten reduzieren,
    # Spalten kanonisch als :tges/:rMenge benennen (die Formel des Fits ist
    # fest auf diese Namen definiert)
    daten_je_teil = Dict{String,DataFrame}()

    for (name, teil) in teile
        name in ausnehmen && continue

        d = teil.df
        if !(hasproperty(d, quelle) && hasproperty(d, prädiktor))
            @warn "lmm_regression!: Teil '$(name)' übersprungen — Spalten $(quelle)/$(prädiktor) fehlen"
            continue
        end

        rows = ausreißer_filtern && hasproperty(d, :Ausreißer) ?
            filter(:Ausreißer => a -> a != "Ausreißer", d) : d

        sub = dropmissing(select(rows, quelle => :tges, prädiktor => :rMenge))
        nrow(sub) == 0 && (@warn "lmm_regression!: Teil '$(name)' ohne verwertbare Punkte"; continue)

        daten_je_teil[name] = sub
    end

    isempty(daten_je_teil) &&
        throw(ArgumentError("lmm_regression!: kein Teil verwertbar — Fit nicht möglich"))

    fit_df = vcat([hcat(sub, DataFrame(Gruppe = fill(name, nrow(sub))))
                   for (name, sub) in daten_je_teil]...)

    # --- 2. globales Modell fitten ------------------------------------------
    lmm = fit(MixedModel,
        @formula(tges ~ 1 + rMenge + (1 + rMenge | Gruppe)),
        fit_df)

    fix = StatsAPI.coef(lmm)            # [β0, β1] — Intercept zuerst
    fix_se = StatsAPI.stderror(lmm)

    # raneftables liefert ein TypedTables.Table (1-D); in einen DataFrame
    # konvertieren: 1. Spalte Gruppenname, 2. rng_Intercept, 3. rng_Slope
    re_df = DataFrame(raneftables(lmm).Gruppe)
    rng = Dict{String,Tuple{Float64,Float64}}(
        string(re_df[i, 1]) => (Float64(re_df[i, 2]), Float64(re_df[i, 3]))
        for i in 1:nrow(re_df))

    # --- 3. je Teil das Gruppen-Ergebnis als RegressionsModell ablegen ------
    gruppen = DataFrame(Gruppe = String[], n_OK = Int[],
                       c_Intercept = Float64[], c_Slope = Float64[],
                       rng_Intercept = Float64[], rng_Slope = Float64[],
                       all_Intercept = Float64[], all_Slope = Float64[])

    for (name, teil) in teile
        name in ausnehmen && continue
        haskey(daten_je_teil, name) || continue

        sub = daten_je_teil[name]
        n_OK = nrow(sub)
        n_Ausreißer = teil.n - n_OK

        r0, r1 = get(rng, name, (NaN, NaN))
        a = fix[1] + r0                  # all_Intercept
        m = fix[2] + r1                   # all_Slope

        x = Float64.(sub.rMenge)
        y = Float64.(sub.tges)
        y_hat = a .+ m .* x

        stats_df = DataFrame(
            Regression = ["MixedModel"],
            LOSS_Fkt = ["ML"],
            n_Rohdaten = teil.n,
            n_OK = n_OK,
            n_Ausreißer = n_Ausreißer,
            pro_Ausreißer = n_Ausreißer / teil.n,
            m = [m],
            n = [a],
            c_Intercept = [fix[1]],
            c_Slope = [fix[2]],
            rng_Intercept = [r0],
            rng_Slope = [r1],
            std_err = [fix_se[2]],
            r2 = [NaN]
        )

        points_df = DataFrame(
            x = x,
            y = y,
            y_hat = y_hat,
            residuals = y .- y_hat
        )

        modell = RegressionsModell(
            name = "MixedModel",
            modell = lmm,
            stats = stats_df,
            points = points_df
        )

        teil.RegressionsModell[modell.name] = modell

        push!(gruppen, (Gruppe = name, n_OK = n_OK,
                        c_Intercept = fix[1], c_Slope = fix[2],
                        rng_Intercept = r0, rng_Slope = r1,
                        all_Intercept = a, all_Slope = m))
    end

    return (lmm = lmm, fix = DataFrame(StatsAPI.coeftable(lmm)), gruppen = gruppen)
end

# =========================================================================
# Stufe 2/3 (Option D): Bewertung und Meta-Regression der Stufe-1-Ergebnisse
# =========================================================================

"""
 # ausschluss_grund(m, a, λ, n_OK, pro_ausreißer; kwargs) -> String

 Prüft, ob das Ergebnis einer Stufe-1-Regression (Koeffizienten einer Gruppe)
 für die Meta-Regression brauchbar ist (Stufe 2 von Option D).

 ## Rückgabe

 `""` wenn brauchbar, sonst der Ausschlussgrund — damit kann gefiltert werden
 mit `filter(:Ausschluss => isempty, ergebnisse)` und der Grund je Gruppe
 mitprotokolliert werden.

 ## Kriterien (Keyword-Argumente)

 - `min_n_OK = 5`: mindestens so viele bereinigte Messpunkte (Theil-Sen
   braucht mehrere unterschiedliche rMenge-Werte, um stabil zu sein).
 - `max_pro_ausreißer = 0.5`: max. Anteil aussortierter Punkte; darüber ist
   die Klassifikation nicht mehr vertrauenswürdig.
 - `positive_steigung = true`: Stückzeit m > 0, negative Werte sind
   physikalisch unsinnig.
 - `positiver_abschnitt = false`: prüft zusätzlich a >= 0. Standard aus, denn
   der Achsenabschnitt extrapoliert auf n = 0 und darf auch bei gutem Fit
   negativ sein, wenn die Daten erst ab n >= 1 beginnen.
"""
function ausschluss_grund(m, a, λ, n_OK, pro_ausreißer;
        min_n_OK::Int = 5,
        max_pro_ausreißer::Float64 = 0.5,
        positive_steigung::Bool = true,
        positiver_abschnitt::Bool = false)

    !isfinite(m) && return "m nicht endlich"
    !isfinite(a) && return "a nicht endlich"
    !isfinite(λ) && return "λ nicht endlich"
    n_OK < min_n_OK && return "n_OK = $(n_OK) < $(min_n_OK)"
    pro_ausreißer > max_pro_ausreißer && return "Ausreißeranteil $(round(pro_ausreißer * 100; digits = 1)) % > $(max_pro_ausreißer * 100) %"
    positive_steigung && m <= 0 && return "Steigung m <= 0"
    positiver_abschnitt && a < 0 && return "Achsenabschnitt a < 0"
    return ""
end

"""
 # meta_regression(daten, ziel; prädiktoren = Symbol[], ...) -> RobustLinearModel

 Meta-Regression (Stufe 3 von Option D): regressiert die Stufe-1-Koeffizienten
 (z.B. `:m` oder `:a` je Gruppe) robust über die Gruppen.

 - Ohne `prädiktoren`: nur Intercept → robust gepoolter Wert (MM) über alle
   Gruppen, funktional das Random-Intercepts-Pendant des LMM.
 - Mit `prädiktoren`: echte Meta-Regression, z.B. `prädiktoren = [:Länge]`
   für `m ~ Länge` (funktional Random-Intercepts und -Slopes).

 Fällt der MM-S-Schritt mit PosDefException (siehe Hinweis in
 `mm_regression_fit`), wird automatisch auf den konvergenten
 M-Estimator (TukeyLoss) ausgewichen.

 ## Beispiel
 ```julia
 meta_m  = meta_regression(AOI_brauchbar, :m)                   # gepoolte Stückzeit
 meta_lm = meta_regression(PTFE_ergebnisse, :m; prädiktoren = [:Länge])
 ```
"""
function meta_regression(daten::AbstractDataFrame, ziel::Symbol;
        prädiktoren::Vector{Symbol} = Symbol[],
        estimator::Type = RobustModels.MMEstimator{RobustModels.TukeyLoss},
        maxiter::Integer = 1000)

    rhs = isempty(prädiktoren) ? StatsModels.term(1) :
                                reduce(+, StatsModels.term.(prädiktoren))
    f = StatsModels.term(ziel) ~ rhs

    try
        return RobustModels.rlm(f, daten, estimator(); σ0 = :mad, maxiter = maxiter)
    catch e
        e isa PosDefException || rethrow()
        @warn "MM-S-Schritt fehlgeschlagen (PosDefException), Fallback auf M-Estimator (TukeyLoss); Ziel: $(ziel)"
        return RobustModels.rlm(f, daten, RobustModels.MEstimator{RobustModels.TukeyLoss}();
                                σ0 = :mad, maxiter = maxiter)
    end
end

"""
 # meta_ausreißer(daten, ziel, modell; k = 3.0) -> DataFrame

 Level-2-Diagnostik der Meta-Regression: Ausreißer-Gruppen. Die Residuen der
 Gruppen gegen die Meta-Regression werden mit der MAD-Skala standardisiert
 (1.4826 * median(|e - median(e)|)); Gruppen mit |r_MAD| > k passen nicht zum
 gepoolten Muster und sind ein genauerer Blick in Stufe 1 wert.

 ## Rückgabe

 DataFrame mit `:Gruppe`, dem Zielwert, Vorhersage, Residuum, `r_MAD` und
 `ist_ausreißer`. Voraussetzung: `daten` enthält eine Spalte `:Gruppe` und
 liegt in derselben Zeilenordnung wie der Fit.
"""
function meta_ausreißer(daten::AbstractDataFrame, ziel::Symbol, modell; k::Float64 = 3.0)
    y = Float64.(daten[!, ziel])
    ŷ = StatsAPI.predict(modell)
    e = y .- ŷ

    σ = 1.4826 * median(abs.(e .- median(e)))
    r_MAD = σ > 0 ? (e .- median(e)) ./ σ : zeros(length(e))

    return DataFrame(
        :Gruppe       => daten.Gruppe,
        ziel          => y,
        :vorhersage   => ŷ,
        :residuum     => e,
        :r_MAD        => r_MAD,
        :ist_ausreißer => abs.(r_MAD) .> k
    )
end