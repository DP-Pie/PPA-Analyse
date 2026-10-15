# =========================================================================
# Hierarchische Regression mit Turing.jl (Weg 2, robustes Mixed-Model-Pendant)
# =========================================================================
#
# Schließt die Lücke von Option D (Stufen 1-3): Dort wird jede Gruppe isoliert
# regressiert; Gruppen mit schwacher Datenlage fliegen raus oder liefern
# unbrauchbare Koeffizienten. Ein Mixed Model nutzt stattdessen ALLE
# Messpunkte und lässt Gruppen mit wenig Daten über die Zufallseffekte von
# den datenreichen Gruppen profitieren (partielles Pooling: die eigenen
# Koeffizienten schrumpfen Richtung Gesamtmuster, umso stärker, je unsicherer
# die Gruppenlage ist).
#
# Dieses Modell kombiniert Pooling und Robustheit in einem Wahrscheinlichkeits-
# modell:
#
#   y_i ~ t(ν, μ_i, σ)           ... Student-t-Fehler, ν geschätzt:
#                                    kleines ν = breite Verteilung, Ausreißer
#                                    werden automatisch heruntergewichtet;
#                                    ν → ∞ entspricht dem normalen LMM.
#   μ_i = α + γ·x_i + (u0_g + u1_g·x_i),  g = Gruppe von i
#   u0, u1 ~ Normal(0, τ), non-zentriert ... Zufalls-Intercept/-Steigung
#
# Nicht-zentrierte Parametrisierung (u = τ · z) und Standardisierung von x
# und y machen das NUTS-Sampling stabil; alle Ergebnisse werden auf die
# Originalskala (Sekunden) zurückgerechnet.
#
# Voraussetzungen (in main.jl vor dem include): using Turing, Distributions,
# MCMCChains, SpecialFunctions.
#
# Laufzeit-Hinweis: Der erste Aufruf kompiliert (Minuten), danach wächst die
# Laufzeit mit Messpunktanzahl × Draws × Ketten. Zum Testen N_samples = 100
# und ketten = 2 verwenden, für die Auswertung 500+.

"""
 # _student_t_logpdf(yg, xg, α, γ, τ0, τ1, z0, z1, σ, ν) -> Float64

 Log-Dichte des hierarchischen Student-t-Modells, summiert über alle
 Beobachtungen. Ausgeschrieben statt über `Distributions.LocationScale`,
 weil sie in jedem Gradientenschritt des Samplers läuft.

 Die Beobachtungen liegen gruppenweise vor (`yg[j]`/`xg[j]` = Werte der
 Gruppe j). Der Koeffizientensatz der Gruppe (a_j = α + τ0·z0[j],
 m_j = γ + τ1·z1[j]) wird einmal als Skalar berechnet und dann über den
 (kleinen) Gruppenvektor gebroadcastet. Das vermeidet die N langen
 Zwischenarrays (μ, z0[g]-Gather, z1[g]-Gather, z, ...) der flachen
 Variante: gemessen ~3x schnellere Gradientenschritte und ~5x weniger
 Allokation, weil die Dual-Zahlen von ForwardDiff bei P Parametern
 jede temporäre Zelle P-fach aufblähen.
"""
function _student_t_logpdf(yg, xg, α, γ, τ0, τ1, z0, z1, σ, ν)
    n = sum(length, yg)
    c = lgamma((ν + 1) / 2) - lgamma(ν / 2) - 0.5 * log(ν * π) - log(σ)
    s = 0.0
    @inbounds for j in eachindex(yg)
        aj = α + τ0 * z0[j]
        mj = γ + τ1 * z1[j]
        s += sum(log1p.(abs2.(yg[j] .- aj .- mj .* xg[j]) ./ (ν * σ * σ)))
    end
    return n * c - ((ν + 1) / 2) * s
end

# --------------------------------------------------------------------------
# Turing-Modell: alle Daten auf der standardisierten Skala
# --------------------------------------------------------------------------
"""
 # hierarchisch_t_modell(yg, xg, G)

 Turing-Modell des hierarchischen Student-t-Regressionsmodells (standardisierte
 Skala, siehe `robust_bayes_lmm` für den Aufruf mit Originaldaten). `yg[j]` und
 `xg[j]` enthalten die standardisierten Beobachtungen der Gruppe j (j in der
 alphabetisch sortierten Gruppenreihenfolge von `robust_bayes_lmm`):

 - `α, γ`: globale Zielgröße bei x = 0 bzw. globale Steigung (Normal(0, 5))
 - `τ0, τ1`: Streuung der Zufalls-Intercepts bzw. -Steigungen (Exponential(1))
 - `z0, z1`: standardisierte Zufallseffekte je Gruppe, non-zentriert
   (Normal(0, 1)); die eigentlichen Effekte sind τ · z
 - `σ`: Reststreuung (Exponential(1))
 - `ν`: Freiheitsgrad der t-Fehler, Prior Exponential(1/30) gestutzt auf [2, 100];
   ν = 2 (Varianz endlich) bis ν = 100 (≈ Normalverteilung)

 Zufalls-Intercepts und -Steigungen sind modellseitig unabhängig voneinander
 (keine Korrelation wie bei (1 + rMenge | Gruppe) im LMM) — die Kovarianz-
 struktur ist eine bewusste Vereinfachung.

 Die Likelihood wird gruppenweise ausgewertet (`_student_t_logpdf`): das ist
 mathematisch identisch zur flachen Variante mit μ_i = α + γ·x_i + τ0·z0[g_i]
 + τ1·z1[g_i]·x_i, aber deutlich schneller (siehe dort).
"""
@model function hierarchisch_t_modell(yg, xg, G)
    α  ~ Normal(0, 5)
    γ  ~ Normal(0, 5)
    τ0 ~ Exponential(1.0)
    τ1 ~ Exponential(1.0)
    z0 ~ filldist(Normal(0, 1), G)
    z1 ~ filldist(Normal(0, 1), G)
    σ  ~ Exponential(1.0)
    ν  ~ truncated(Exponential(1 / 30), 2.0, 100.0)

    lp = _student_t_logpdf(yg, xg, α, γ, τ0, τ1, z0, z1, σ, ν)
    Turing.@addlogprob! isfinite(lp) ? lp : -Inf
end

# --------------------------------------------------------------------------
# Wrapper: Datenaufbereitung, Sampling, Rückrechnung auf Originalskala
# --------------------------------------------------------------------------
"""
 # robust_bayes_lmm(daten, yvar, xvar, gruppe; N_samples = 500, ketten = 4, seed) -> NamedTuple

 Fit des hierarchischen Student-t-Modells (Weg 2) über ALLE Messpunkte:
 partielles Pooling je Gruppe (`gruppe`) plus robuste Fehler.

 ## Ablauf

 1. x und y standardisieren (Stabilität des Samplers) und die Gruppen in eine
    stabile, alphabetisch sortierte Reihenfolge bringen.
 2. NUTS-Sampling (non-zentrierte Zufallseffekte). Mit mehreren Julia-Threads
    laufen die Ketten parallel (MCMCThreads), sonst sequenziell (MCMCSerial).
    Mit `kette = chn` kann eine bereits gesampelte Kette (gleicher Datenstand!)
    übergeben werden, dann wird das Sampling übersprungen — praktisch zum
    Iterieren an der Nachbearbeitung. Akzeptiert werden das Chain-Objekt
    selbst oder der Breitformat-DataFrame der Draws (eine Spalte je Parameter,
    via `DataFrame(chn)` erzeugt und z.B. per JLD2 gesichert); beim DataFrame
    entfällt R-hat (Spalte bleibt NaN).
 3. Rückrechnung aller Posterior-Größen auf die Originalskala (Sekunden).

 ## Rückgabe

 NamedTuple `(kette, fix, gruppe, ν, σ)`:
 - `kette::Chains` ... die volle Posterior-Kette (für eigene Auswertungen)
 - `fix::DataFrame` ... globale Koeffizienten und Modellgrößen mit 95 %-Intervall
   und R-hat: `m` (Stückzeit [s/Stück]), `a` (Achsenabschnitt [s]),
   `σ` (Reststreuung), `τ0`/`τ1` (Streuung der Gruppen-Koeffizienten),
   `ν` (t-Freiheitsgrade; klein ⇒ Ausreißer im Datenbestand)
 - `gruppe::DataFrame` ... partiell gepoolte Koeffizienten je Gruppe:
   `a`, `m` mit 95 %-Intervall, `n_obs` und der Gruppenname
 - `ν`, `σ` ... Posterior-Mittel als Skalar

 ## Diagnostik

 - R-hat in `fix` sollte ≈ 1 sein; deutlich darüber: `N_samples` erhöhen.
 - `τ1` nahe 0: die Steigungen unterscheiden sich zwischen Gruppen kaum, ein
   Random-Slope lohnt dann nicht. Dasselbe gilt für `τ0` und die Abschnitte.
 - Kleines ν (z. B. < 10): Ausreißer sind vorhanden und werden über die
   t-Verteilung heruntergewichtet (genau der gewünschte Effekt).

 ## Beispiel
 ```julia
 erg = robust_bayes_lmm(AOI_fit_df, :Zeit, :rMenge, :Gruppe)
 vscodedisplay(erg.fix)
 vscodedisplay(erg.gruppe)
 ```
"""
function robust_bayes_lmm(daten::AbstractDataFrame, yvar::Symbol, xvar::Symbol, gruppe::Symbol;
        N_samples::Int = 500,
        ketten::Int = 4,
        seed::Int = 20261013,
        kette = nothing)

    daten = dropmissing(DataFrame(daten), [yvar, xvar, gruppe])

    y = Float64.(daten[!, yvar])
    x = Float64.(daten[!, xvar])
    μy, sy = mean(y), std(y)
    μx, sx = mean(x), std(x)
    sy > 0 || throw(ArgumentError("Zielgröße :$yvar ist konstant."))
    sx > 0 || throw(ArgumentError("Prädiktor :$xvar ist konstant."))

    ys = (y .- μy) ./ sy
    xs = (x .- μx) ./ sx

    gruppen_namen = sort(unique(String.(daten[!, gruppe])))
    G = length(gruppen_namen)
    G >= 2 || throw(ArgumentError("Partielles Pooling braucht mindestens 2 Gruppen, gefunden: $(G)"))

    # Beobachtungen einmal gruppenweise einsortieren (O(N), kein erneutes
    # Suchen je Gruppe) — das Modell konsumiert genau diese Reihenfolge.
    g_idx = Dict(name => j for (j, name) in enumerate(gruppen_namen))
    yg = [Float64[] for _ in 1:G]
    xg = [Float64[] for _ in 1:G]
    for (i, name) in enumerate(String.(daten[!, gruppe]))
        j = g_idx[name]
        push!(yg[j], ys[i])
        push!(xg[j], xs[i])
    end
    n_obs = length.(yg)

    println("Weg 2 (Bayes): N = $(length(y)) Messpunkte, G = $(G) Gruppen, " *
            "$(ketten) Ketten × $(N_samples) Draws (Julia-Threads: $(Threads.nthreads()))")

    if kette === nothing
        Random.seed!(seed)
        modell = hierarchisch_t_modell(yg, xg, G)
        mcmcsampler = Threads.nthreads() > 1 ? MCMCThreads() : MCMCSerial()
        chn = Turing.sample(modell, NUTS(), mcmcsampler, N_samples, ketten;
                            progress = true)
    else
        # Bereits gesampelte Kette wiederverwenden (gleicher Datenstand!),
        # um die Nachbearbeitung ohne erneutes Sampling zu iterieren.
        chn = kette
    end

    # Die komplette Nachbearbeitung läuft in try/catch: Schlägt sie fehl
    # (z.B. API-Unterschiede der MCMCChains-Version), bleibt die bereits
    # gesampelte Kette erhalten und kann via `kette = ...` ohne erneutes
    # Sampling weiterverarbeitet werden.
    try
    # ---- Posterior auf Originalskala zurückrechnen ------------------------
    # Extraktion über den Tables-Weg (DataFrame der Kette), da die klassische
    # Indexierung chn[:, name, :] in neueren MCMCChains-Versionen (FlexiChains)
    # entfernt wurde. Eine Zeile = ein Draw (Iteration × Kette); Skalar- und
    # Vektor-Parameter stehen damit automatisch draw-weise nebeneinander.
    draws = chn isa AbstractDataFrame ? chn : DataFrame(chn)
    if hasproperty(draws, :parameter) && hasproperty(draws, :value)
        # Langformat (iteration, chain, parameter, value) → Breitformat
        draws = unstack(draws, [:chain, :iteration], :parameter, :value)
    end

    _hole(s) = begin
        sp = Symbol(s)
        hasproperty(draws, sp) ||
            throw(ArgumentError("Parameter '$(s)' nicht in der Kette gefunden; " *
                                "vorhandene Spalten: $(String.(propertynames(draws)))"))
        return Float64.(draws[!, sp])
    end

    α_d  = reshape(_hole("α"), :, 1)      # D × 1, für Broadcasting mit (D × G)
    γ_d  = reshape(_hole("γ"), :, 1)
    τ0_d = reshape(_hole("τ0"), :, 1)
    τ1_d = reshape(_hole("τ1"), :, 1)
    σ_d  = _hole("σ")
    ν_d  = _hole("ν")
    z0_d = reduce(hcat, [_hole("z0[$(j)]") for j in 1:G])   # D × G
    z1_d = reduce(hcat, [_hole("z1[$(j)]") for j in 1:G])

    # Steigung m und Achsenabschnitt a je Gruppe (Originalskala):
    #   m_g = sy * (γ + τ1·z1_g) / sx
    #   a_g = μy + sy * (α + τ0·z0_g) - m_g * μx
    m_draws = (sy / sx) .* (γ_d .+ z1_d .* τ1_d)              # D × G
    a_draws = μy .+ sy .* (α_d .+ z0_d .* τ0_d) .- m_draws .* μx

    m_glob_d = sy .* γ_d ./ sx
    a_glob_d = μy .+ sy .* α_d .- m_glob_d .* μx

    # R-hat aus summarystats je Parameter; schlägt die Extraktion fehl
    # (API-Unterschiede zwischen MCMCChains-Versionen), bleibt R_hat NaN und
    # stört die restliche Auswertung nicht.
    rhats = Dict{String, Float64}()
    if !(chn isa AbstractDataFrame)   # DataFrame-Übergabe: kein R-hat verfügbar
    try
        ss_df = DataFrame(MCMCChains.summarystats(chn))
        p_spalte = hasproperty(ss_df, :parameters) ? :parameters :
                   hasproperty(ss_df, :param)      ? :param      : nothing
        if p_spalte !== nothing && hasproperty(ss_df, :rhat)
            for i in 1:nrow(ss_df)
                rhats[string(ss_df[!, p_spalte][i])] = Float64(ss_df.rhat[i])
            end
        end
    catch e
        @warn "R-hat konnte nicht extrahiert werden ($(typeof(e))) — Spalte R_hat bleibt NaN"
    end
    end

    # vec(): die Skalar-Größen liegen als D×1-Matrix vor (nötig für das
    # Broadcasting mit den D×G-Zufallseffekten); quantile braucht einen Vektor.
    _mw(v) = mean(v)
    _lo(v) = quantile(vec(v), 0.025)
    _hi(v) = quantile(vec(v), 0.975)

    fix = DataFrame(
        :Parameter  => ["m (Stückzeit)", "a (Achsenabschnitt)",
                        "σ (Reststreuung)", "τ0 (Streuung a je Gruppe)",
                        "τ1 (Streuung m je Gruppe)", "ν (t-Freiheitsgrade)"],
        :Skala      => ["s/Stück", "s", "s", "s", "s/Stück", "-"],
        :Mittelwert => [_mw(m_glob_d), _mw(a_glob_d), _mw(sy .* σ_d),
                        _mw(sy .* τ0_d), _mw(sy .* τ1_d ./ sx), _mw(ν_d)],
        :Q2_5       => [_lo(m_glob_d), _lo(a_glob_d), _lo(sy .* σ_d),
                        _lo(sy .* τ0_d), _lo(sy .* τ1_d ./ sx), _lo(ν_d)],
        :Q97_5      => [_hi(m_glob_d), _hi(a_glob_d), _hi(sy .* σ_d),
                        _hi(sy .* τ0_d), _hi(sy .* τ1_d ./ sx), _hi(ν_d)],
        :R_hat      => [get(rhats, "γ", NaN), get(rhats, "α", NaN),
                        get(rhats, "σ", NaN), get(rhats, "τ0", NaN),
                        get(rhats, "τ1", NaN), get(rhats, "ν", NaN)]
    )

    gruppe_df = DataFrame(
        :Gruppe => gruppen_namen,
        :n_obs  => n_obs,
        :a      => vec(mean(a_draws, dims = 1)),
        :a_lo   => [quantile(a_draws[:, j], 0.025) for j in 1:G],
        :a_hi   => [quantile(a_draws[:, j], 0.975) for j in 1:G],
        :m      => vec(mean(m_draws, dims = 1)),
        :m_lo   => [quantile(m_draws[:, j], 0.025) for j in 1:G],
        :m_hi   => [quantile(m_draws[:, j], 0.975) for j in 1:G]
    )

    return (kette = chn, fix = fix, gruppe = gruppe_df,
            skala = (μy = μy, sy = sy, μx = μx, sx = sx, gruppen = gruppen_namen),
            ν = mean(ν_d), σ = mean(sy .* σ_d))
    catch e
        @error "Nachbearbeitung fehlgeschlagen ($(typeof(e)): $(e)). " *
               "Die gesampelte Kette bleibt erhalten und kann mit " *
               "robust_bayes_lmm(...; kette = <kette>) ohne erneutes " *
               "Sampling weiterverarbeitet werden."
        return (kette = chn, fix = nothing, gruppe = nothing,
                skala = (μy = μy, sy = sy, μx = μx, sx = sx, gruppen = gruppen_namen),
                ν = NaN, σ = NaN)
    end
end

# =========================================================================
# Validierung des Bayes-Modells (Weg 2)
# =========================================================================
#
# Drei Ebenen, analog zur klassischen Diagnostik-Pipeline:
#   1. bayes_mcmc_diagnostik  ... Konvergenz des Samplers über ALLE Parameter
#      (auch die Zufallseffekte z0/z1 — ein schlechter R-hat dort macht die
#      Gruppen-Koeffizienten unbrauchbar, selbst wenn α/γ sauber sind)
#   2. bayes_residuen         ... Punktgenaue Residuen (Posterior-Mittel der
#      Vorhersage je Messpunkt) für Trichter-/QQ-Plots
#   3. bayes_abdeckung        ... liegen die beobachteten Punkte im 95 %-Band
#      der t(ν)-Fehler? Je Gruppe, mit RMSE und Median-Residuum

"""
 # bayes_mcmc_diagnostik(bayes; rhat_grenze = 1.01, ess_grenze = 100) -> DataFrame

 Konvergenz-Diagnostik des Samplers über ALLE Parameter der Kette, inklusive
 der Zufallseffekte z0[j]/z1[j] je Gruppe — die fix-Tabelle aus
 `robust_bayes_lmm` prüft nur α, γ, σ, τ0, τ1, ν.

 ## Rückgabe

 DataFrame mit `:Parameter`, `:R_hat`, `:ESS` und `:OK` (Kriterium:
 R-hat <= `rhat_grenze` und ESS >= `ess_grenze`; fehlende Werte gelten als
 unauffällig). Fehlt eine Größe in summarystats (Versionsunterschied), bleibt
 sie NaN und stört nicht.

 Wurde die Kette als DataFrame übergeben (via `kette = DataFrame(chn)`), gibt
 es keine MCMC-Diagnostik — dann nur eine Warnung und ein leeres DataFrame.
"""
function bayes_mcmc_diagnostik(bayes; rhat_grenze::Float64 = 1.01, ess_grenze::Float64 = 100.0)
    chn = bayes.kette
    if chn isa AbstractDataFrame
        @warn "Kette als DataFrame übergeben — keine MCMC-Diagnostik (R-hat/ESS) möglich."
        return DataFrame()
    end

    ss = DataFrame(MCMCChains.summarystats(chn))
    p_spalte = hasproperty(ss, :parameters) ? :parameters :
               hasproperty(ss, :param)      ? :param      : nothing
    p_spalte === nothing && throw(ArgumentError("Parameter-Spalte in summarystats nicht gefunden: $(propertynames(ss))"))

    rhatspalte = hasproperty(ss, :rhat) ? ss.rhat :
                 hasproperty(ss, :Rhat) ? ss.Rhat : fill(NaN, nrow(ss))
    essspalte  = hasproperty(ss, :ess_bulk) ? ss.ess_bulk :
                 hasproperty(ss, :n_eff)    ? ss.n_eff    : fill(NaN, nrow(ss))

    rhat = [coalesce(ismissing(v) ? NaN : Float64(v), NaN) for v in rhatspalte]
    ess  = [coalesce(ismissing(v) ? NaN : Float64(v), NaN) for v in essspalte]

    ok = [(isnan(r) || r <= rhat_grenze) && (isnan(e) || e >= ess_grenze)
          for (r, e) in zip(rhat, ess)]

    out = DataFrame(
        :Parameter => string.(ss[!, p_spalte]),
        :R_hat => rhat,
        :ESS   => ess,
        :OK    => ok
    )
    n_schlecht = count(.!ok)
    println("MCMC-Diagnostik: $(nrow(out)) Parameter, $(n_schlecht) auffällig " *
            "(Kriterium: R-hat <= $(rhat_grenze), ESS >= $(ess_grenze))")
    return out
end

"""
 # bayes_residuen(daten, yvar, xvar, gruppe, bayes) -> DataFrame

 Punktgenaue Residuen des Bayes-Modells: Vorhersage je Messpunkt als
 Posterior-Mittel über alle Draws (inkl. Zufallseffekte der zugehörigen
 Gruppe), zurückgerechnet auf die Originalskala.

 ## Rückgabe

 DataFrame im Zeilen-Format der Daten mit `:Gruppe`, `:x`, `:y`,
 `:vorhersage`, `:residuum` und `:resid_std` (Residuum geteilt durch die
 Posterior-Streuung σ; für QQ-Plots gegen t(ν) mit ν aus `bayes.ν`).

 ## Voraussetzung

 `bayes` muss von der aktuellen `robust_bayes_lmm`-Version stammen (enthält
 `bayes.skala` mit den Standardisierungsgrößen). Nach dem Neuladen der Datei
 genügt `robust_bayes_lmm(...; kette = <gesicherte Kette>)` — kein neues
 Sampling nötig.

 ## Keyword `mitnehmen`

 Zusätzliche Spalten aus `daten`, die unverändert in das Ergebnis übernommen
 werden (z.B. `mitnehmen = [:MA, :Arbeiter, :RMNr]`) — zum Färben der
 Residuen-Plots und Aufklären von Strängen in der Residuen-Wolke.
"""
function bayes_residuen(daten::AbstractDataFrame, yvar::Symbol, xvar::Symbol,
                        gruppe::Symbol, bayes; mitnehmen::Vector{Symbol} = Symbol[])

    skala = bayes.skala
    draws = bayes.kette isa AbstractDataFrame ? bayes.kette : DataFrame(bayes.kette)

    _hole(s) = begin
        sp = Symbol(s)
        hasproperty(draws, sp) ||
            throw(ArgumentError("Parameter '$(s)' nicht in der Kette gefunden; " *
                                "vorhandene Spalten: $(String.(propertynames(draws)))"))
        return Float64.(draws[!, sp])
    end

    daten = dropmissing(DataFrame(daten), [yvar, xvar, gruppe])
    y = Float64.(daten[!, yvar])
    x = Float64.(daten[!, xvar])

    g = [findfirst(==(v), skala.gruppen) for v in String.(daten[!, gruppe])]
    fehlend = unique(String.(daten[!, gruppe])[findall(isnothing, g)])
    isempty(fehlend) ||
        throw(ArgumentError("Diese Gruppen sind in den Daten, aber nicht im Modell-Fit: " *
                            "$(join(fehlend, ", "))"))

    G = length(skala.gruppen)
    α_d  = reshape(_hole("α"), :, 1)
    γ_d  = reshape(_hole("γ"), :, 1)
    τ0_d = reshape(_hole("τ0"), :, 1)
    τ1_d = reshape(_hole("τ1"), :, 1)
    z0_d = reduce(hcat, [_hole("z0[$(j)]") for j in 1:G])
    z1_d = reduce(hcat, [_hole("z1[$(j)]") for j in 1:G])

    # Vorhersage je Draw × Beobachtung auf standardisierter Skala,
    # dann Posterior-Mittel je Beobachtung und Rückrechnung in Sekunden
    xs_row = reshape((x .- skala.μx) ./ skala.sx, 1, :)
    μ_std  = α_d .+ γ_d .* xs_row .+ z0_d[:, g] .+ z1_d[:, g] .* xs_row
    fitted = skala.μy .+ skala.sy .* vec(mean(μ_std, dims = 1))

    resid = y .- fitted

    # σ-Proxy, falls die Nachbearbeitung fehlgeschlagen war (σ = NaN):
    # MAD-Skala der Residuen als robuste Notfallgröße
    σ = isfinite(bayes.σ) ? bayes.σ : 1.4826 * median(abs.(resid .- median(resid)))

    return DataFrame(
        :Gruppe     => daten[!, gruppe],
        :x          => x,
        :y          => y,
        :vorhersage => fitted,
        :residuum   => resid,
        :resid_std  => resid ./ σ
    ) |> out -> begin
        # Zusatzspalten aus mitnehmen übernehmen (gleiche Zeilenauswahl)
        for c in mitnehmen
            hasproperty(daten, c) ||
                throw(ArgumentError("mitnehmen-Spalte :$c fehlt in daten"))
            out[!, c] = daten[!, c]
        end
        out
    end
end

"""
 # _abdeckung_bewertung(n, abd, niveau, defizit_se, bias_std, trend_std, streu_faktor) -> (Int, String)

 Stufe und Begründungstext einer Gruppe in `bayes_abdeckung` — analog zu
 `bewerte_diagnostik!` in der klassischen Pipeline: Jede Dimension liefert
 (Stufe, Text), die finale Stufe ist das Minimum über alle Dimensionen, und
 die Begründung nennt nur die Dimension(en), die tatsächlich bindend sind
 (nicht-bindende schweigen still).

 Stufenwert: 1 = Verwerfen … 5 = Sehr sicher (siehe `ABDECKUNG_STUFEN`).

 - `n` ... Messpunkte der Gruppe
 - `defizit_se` ... (niveau − Abdeckung) / Standardfehler der Binomial-
   Abdeckung; wie viele Standardfehler die Abdeckung unter dem Soll liegt.
   n-korrigiert: kleine Gruppen tolerieren mehr Abdeckungsrauschen.
 - `bias_std` ... |median(resid_std)| — systematische Fehlprognose in σ
 - `trend_std` ... Differenz der Median-Residuen zwischen oberer und unterer
   Mengenhälfte, in σ — nichtlineare Mengenabhängigkeit (die im Median
   wegmittelt und von bias_std/Abdeckung unbemerkt bleibt); NaN, wenn zu
   wenig Punkte je Hälfte, dann greift das Kriterium nicht.
 - `streu_faktor` ... robuste Streuung der Gruppe / robuste Streuung gesamt;
   > 1 heißt: die Gruppe ist lauter, als das Modell für sie annimmt
"""
const ABDECKUNG_STUFEN = ("Verwerfen", "Untersuchen", "Akzeptabel", "Sicher", "Sehr sicher")

function _abdeckung_bewertung(n, abd, niveau, defizit_se, bias_std, trend_std, streu_faktor)
    dim = Tuple{Int, String}[]

    # Deckung: 1/2/3/6-SE-Leiter
    s = defizit_se > 6 ? 1 : defizit_se > 3 ? 2 : defizit_se > 2 ? 3 : defizit_se > 1 ? 4 : 5
    push!(dim, (s, s < 5 ?
        "Abdeckung $(round(Int, 100 * abd)) % statt $(round(Int, 100 * niveau)) % " *
        "($(round(defizit_se, digits = 1)) SE zu niedrig)" : ""))

    # Systematischer Bias in σ
    s = bias_std > 1.0 ? 1 : bias_std > 0.5 ? 2 : bias_std > 0.3 ? 3 : bias_std > 0.15 ? 4 : 5
    push!(dim, (s, s < 5 ? "Median-Residuum $(round(bias_std, digits = 2))σ" : ""))

    # Trend über die Menge in σ (NaN ⇒ Kriterium deaktiviert)
    s = isnan(trend_std) ? 5 :
        trend_std > 1.0 ? 2 : trend_std > 0.6 ? 3 : trend_std > 0.3 ? 4 : 5
    push!(dim, (s, s < 5 ? "Residuum-Trend über Menge $(round(trend_std, digits = 2))σ" : ""))

    # Streuung relativ zum Gesamtmodell
    s = streu_faktor > 2.0 ? 2 : streu_faktor > 1.5 ? 3 : streu_faktor > 1.25 ? 4 : 5
    push!(dim, (s, s < 5 ?
        "Streuung $(round(streu_faktor, digits = 1))× so hoch wie gesamt" : ""))

    # Aussagekraft: mit < 10 Punkten keine belastbare Aussage möglich
    n < 10 && push!(dim, (2, "n = $(n), zu wenig für eine belastbare Aussage"))

    stufe = isempty(dim) ? 5 : minimum(s for (s, _) in dim)
    grund = join([g for (s, g) in dim if s == stufe && !isempty(g)], "; ")
    return (stufe, grund)
end

"""
 # bayes_abdeckung(residuen; ν = 30, niveau = 0.95) -> DataFrame

 Güte je Gruppe: Liegt der erwartete Anteil der Punkte im Vorhersageband der
 t(ν)-Fehler? Bandbreite = t-Quantil bei `niveau` mal standardisiertes
 Residuum. Daneben RMSE und Median/MAD der Residuen je Gruppe.

 ## Einstufung

 Je Gruppe werden fünf Kriterien berechnet und zu einer Stufe zusammenge-
 führt (siehe `_abdeckung_bewertung` für die Regeln). Die Spalte `:Begrün-
 dung` nennt — wie `bewerte_diagnostik!` in der klassischen Pipeline — nur
 die Kriterien, die tatsächlich bindend sind (nicht-bindende schweigen):

 | Stufe | Bedeutung |
 |---|---|
 | Verwerfen | Abdeckung > 6 SE unter Soll oder Bias > 1σ — das Modell passt für diese Gruppe nicht; Ursache klären (Restruktur, nichtlineare Mengenabhängigkeit, Unterbestückung), nicht blind übernehmen |
 | Untersuchen | Deutlich auffällig (> 3 SE / Bias > 0.5σ / Trend > 1σ / doppelt so laut) oder n < 10 — manuell ansehen |
 | Akzeptabel | Leichte Auffälligkeiten (> 2 SE / Bias > 0.3σ / Trend > 0.6σ / 1.5-fache Streuung) — brauchbar, aber im Blick behalten |
 | Sicher | Kleinere Abweichungen (> 1 SE / Bias > 0.15σ / Trend > 0.3σ / 1.25-fache Streuung) |
 | Sehr sicher | alles unauffällig — Gruppe folgt dem gepoolten Muster |

 Die SE-Grenzen sind bewusst als 1/2/3/6-Leiter gewählt: eine tatsächlich
 saubere Gruppe überschreitet 1 SE rein zufällig in ~16 % der Fälle — sie
 landet dann bei "Sicher", nicht schlechter.

 Das Ergebnis ist nach Schwere sortiert (schlechteste Gruppe zuerst), damit
 der Blick sofort auf die Problemgruppen fällt.

 ## Interpretation

 - Abdeckung deutlich unter `niveau` (z. B. 0.85 statt 0.95): das Modell ist
   für diese Gruppe zu sicher — Restruktur vorhanden (Heteroskedastizität,
   nichtlineare Mengenabhängigkeit oder Underfitting).
 - Median-Residuum systematisch ≠ 0: systematische Fehlprognose der Gruppe.
 - "Verwerfen" heißt nicht, dass die Messpunkte falsch sind — nur dass das
   gemeinsame Modell (Pooling + t-Fehler) diese Gruppe nicht abbildet.
"""
function bayes_abdeckung(residuen::AbstractDataFrame; ν::Real = 30.0, niveau::Float64 = 0.95)
    ν = isfinite(ν) ? ν : 30.0
    t_q = quantile(TDist(ν), 0.5 + niveau / 2)
    abd_spalte = Symbol("abdeckung_", round(Int, niveau * 100))

    # trend_std braucht :x (aus bayes_residuen immer vorhanden); ohne :x wird
    # das Kriterium deaktiviert (NaN) statt zu werfen.
    if !hasproperty(residuen, :x)
        residuen = copy(residuen)
        residuen[!, :x] = fill(NaN, nrow(residuen))
    end

    out = combine(groupby(residuen, :Gruppe),
        nrow => :n,
        :resid_std => (r -> mean(abs.(r) .<= t_q)) => abd_spalte,
        :resid_std => (r -> abs(median(r))) => :bias_std,
        [:x, :resid_std] => ((x, r) -> begin
            lo = x .< median(x); hi = .!lo
            (count(lo) >= 5 && count(hi) >= 5) ? median(r[hi]) - median(r[lo]) : NaN
        end) => :trend_std,
        :resid_std => (r -> 1.4826 * median(abs.(r .- median(r)))) => :_mad_std,
        :residuum  => (r -> sqrt(mean(abs2.(r)))) => :RMSE,
        :residuum  => median => :median_resid,
        :residuum  => (r -> 1.4826 * median(abs.(r .- median(r)))) => :MAD_resid
    )

    # Referenzstreuung über ALLE Punkte, beide MADs auf der standardisierten
    # Skala (σ-Einheiten), damit der Faktor skalenfrei ist:
    # streu_faktor = Gruppen-MAD / Gesamt-MAD > 1 ⇒ Gruppe lauter, als das
    # Gesamtmodell für sie annimmt. MAD_resid (Sekunden) bleibt nur zur Anzeige.
    r_std = Float64.(residuen.resid_std)
    mad_global = 1.4826 * median(abs.(r_std .- median(r_std)))
    streu = mad_global > 0 ? out._mad_std ./ mad_global : fill(Inf, nrow(out))

    defizit = [(niveau - a) / sqrt(niveau * (1 - niveau) / n)
               for (a, n) in zip(out[!, abd_spalte], out.n)]
    out[!, :deckung_defizit_se] = round.(defizit; digits = 2)
    out[!, :streu_faktor] = round.(streu; digits = 2)
    bewertung = [_abdeckung_bewertung(n, a, niveau, d, b, t, s)
                 for (n, a, d, b, t, s) in zip(out.n, out[!, abd_spalte], defizit,
                                               out.bias_std, out.trend_std, streu)]
    out[!, :Einstufung] = [ABDECKUNG_STUFEN[w] for (w, _) in bewertung]
    out[!, :Begründung] = [g for (_, g) in bewertung]

    # Schlechteste Gruppe zuerst
    _rang = Dict(s => i for (i, s) in enumerate(ABDECKUNG_STUFEN))
    sort!(out, [:Einstufung]; by = s -> _rang[s])

    println("Abdeckung gesamt: $(round(100 * mean(abs.(residuen.resid_std) .<= t_q); digits = 1)) % " *
            "(Soll: $(round(Int, niveau * 100)) %)")
    println("Einstufung: " * join(["$(count(==(s), out.Einstufung))× $s"
            for s in ("Verwerfen", "Untersuchen", "Akzeptabel", "Sicher", "Sehr sicher")
            if count(==(s), out.Einstufung) > 0], ", "))
    return select(out, :Gruppe, :n, :Einstufung,
                 abd_spalte, :deckung_defizit_se, :bias_std, :trend_std, :streu_faktor,
                 :RMSE, :median_resid, :MAD_resid, :Begründung)
end
