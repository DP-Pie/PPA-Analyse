# =========================================================================
# Ausreißer-Flag-System (Bitmasken-Klassifikation)
# =========================================================================
# Bit-Belegung des Ausreißer-Bytes
const FLAG_RSTUDENT = 0b0001   # 2^0
const FLAG_HEBEL    = 0b0010   # 2^1
const FLAG_COOKS    = 0b0100   # 2^2
const FLAG_UKZ      = 0b1000   # 2^3
const FLAG_RSTUDENT3 = 0b10000 # 2^4

"""
 # ist_ausreißer(flags::Integer) -> Bool

 Explizite Entscheidungsregel aus dem Ausreißer-Byte:

- Ultrakurzzeit (Bit 3) gesetzt → Ausreißer (Domänenregel, greift allein).
- RStudent_extrem (Bit 4) gesetzt → Ausreißer (|RStudent| weit über dem
  Schwellenwert, greift allein).
- Alle drei Diagnose-Bits (RStudent, Hebelwert, CooksAbstand) gesetzt → Ausreißer.
"""
ist_ausreißer(flags::Integer) = (flags & (FLAG_UKZ | FLAG_RSTUDENT3)) != 0 ||
    (flags & (FLAG_RSTUDENT | FLAG_HEBEL | FLAG_COOKS)) == (FLAG_RSTUDENT | FLAG_HEBEL | FLAG_COOKS)

"""
 # ausreißer_flags(modell, daten::DataFrame; kwargs...)

 Regelbasierte Ausreißerklassifikation über fünf Flags (vier Diagnose-Flags plus
 Ultrakurzzeit), die als
 Bitmaske zusammengefasst werden:

 | Bit | Flag          | Kriterium                          |
 |-----|---------------|-----------------------------------|
 | 0   | `flag_rstudent` | \\|RStudent\\| > `t_rstudent` (Standard 2.5) |
 | 1   | `flag_hebel`    | hᵢᵢ > `hebel_faktor` * p / n (Standard: 2/n) |
 | 2   | `flag_cooks`    | CooksAbstand > `t_cooks` / n (Standard: 4/n) |
 | 3   | `flag_ukz`      | Ultrakurzzeit: rMenge > `ukz_n` und tges < `ukz_tges` |
 | 4   | `flag_rstudent3` | \\|RStudent\\| > `rstudent_faktor` * `t_rstudent` (Standard: 3 * 2.5 = 7.5) |

 Entscheidungsregel:
 - `flag_ukz == 1`  → sicherer Ausreißer, da Fehler beim Abstempels des Produktionsauftrags (Domänenregel, OR-Override).
 - `flag_rstudent3 == 1` → Ausreißer: Das Residuum liegt um den Faktor
   `rstudent_faktor` über dem RStudent-Schwellenwert und ist damit so extrem,
   dass es keines weiteren Kriteriums bedarf (OR-Override).
 - `flag_rstudent == flag_hebel == flag_cooks == 1` → Ausreißer
   (alle drei Diagnose-Flags gemeinsam kritisch, AND-Verknüpfung nach
   dem klassischen Schema der Regressionsdiagnostik).
 - mindestens `verdacht_min` der drei Diagnose-Flags (Standard: 2, Mehrheitsvotum)
   → Verdacht: keine harte Ausschlussentscheidung, aber eine Kontrolle wert.

 Die Bitmaske `flags` ist eine **ID** (welche Kriterien erfüllt sind), keine
 Ordinalskala — ihr Zahlenwert darf nicht als "Schweregrad" verglichen werden.
 Dafür gibt es `n_diagnose` (Anzahl erfüllter Diagnose-Flags, 0–3) und `einstufung`
 ("OK" / "Verdacht" / "Ausreißer"). Die harten Alleinkriterien `flag_ukz` und
 `flag_rstudent3` zählen nicht in `n_diagnose`, da sie allein schon zum
 Ausreißer machen.

 Gibt einen DataFrame zurück mit den Rohwerten (`rstudent`, `hii`, `cooksdist`),
 den einzelnen Flags, der Bitmaske `flags`, der Anzahl `n_diagnose`, der
 Einstufung `einstufung` und dem Endergebnis `ist_ausreißer`.

 ## Keyword-Argumente
 - `t_rstudent::Float64 = 2.5` — Grenzwert für den extern studentisierten Residualwert (Absolutbetrag).
 - `rstudent_faktor::Float64 = 3.0` — Faktor für die harte RStudent-Regel:
   \\|RStudent\\| > `rstudent_faktor` * `t_rstudent` setzt `flag_rstudent3`
   und macht den Punkt allein dadurch zum Ausreißer.
 - `p::Int64 = 2` — Anzahl der Modellparameter (einfache lineare Regression: Achsenabschnitt + Steigung).
 - `hebel_faktor::Float64 = 1.0` — Faktor vor p/n für den Hebelwert-Grenzwert. Standard 1.0 → h̄ = 2/n.
   (Lehrbuchüblich ist auch 2.0, also h̄ = 2p/n = 4/n — strengerer Cut.)
 - `t_cooks::Float64 = 4.0` — CooksAbstand-Grenzwert ist `t_cooks / n` (Standard 4/n).
 - `ukz_n::Float64 = 1.0` — Ultrakurzzeit-Kriterium: rMenge > ukz_n.
 - `ukz_tges::Float64 = 200.0` — Ultrakurzzeit-Kriterium: tges < ukz_tges.
 - `verdacht_min::Int64 = 2` — ab so vielen erfüllten Diagnose-Flags gilt der Punkt als "Verdacht".

 `daten` ist der DataFrame, aus dem das Regressions-Setting erstellt wurde
 (er liefert n, rMenge und tges).
"""
function ausreißer_flags(modell, daten::DataFrame;
    t_rstudent::Float64 = 2.5,
    rstudent_faktor::Float64 = 3.0,
    p::Int64 = 2,
    hebel_faktor::Float64 = 1.0,
    t_cooks::Float64 = 4.0,
    ukz_n::Float64 = 1.0,
    ukz_tges::Float64 = 200.0,
    verdacht_min::Int64 = 2
)
    # Diagnosewerte aus dem Modell
    rstudent = studentizedResiduals(modell)
    hii = diag(hatmatrix(modell))
    cooksdist = cooks(modell)

    return _ausreißer_flags(daten, rstudent, hii, cooksdist;
        t_rstudent = t_rstudent,
        rstudent_faktor = rstudent_faktor,
        p = p,
        hebel_faktor = hebel_faktor,
        t_cooks = t_cooks,
        ukz_n = ukz_n,
        ukz_tges = ukz_tges,
        verdacht_min = verdacht_min
    )
end

"""
 # _ausreißer_flags(daten, rstudent, hii, cooksdist; kwargs...) -> DataFrame

 Kern der Flag-Berechnung aus fertigen Diagnosegrößen. `ausreißer_flags`
 (klassisches Modell) und `ausreißer_flags_setzen!` (klassisch bzw. robust
 über die Registry `REG_CORES`) rufen diese Funktion mit ihren jeweiligen
 Diagnosewerten auf und bekommen denselben Auswertungs-DataFrame zurück.

 Fehlende Diagnosewerte (`missing`, z.B. `rstudent_mad`/`cooks_mad` bei
 MAD-Skala 0) gelten als nicht kritisch und lösen kein Flag aus.
"""
function _ausreißer_flags(daten::DataFrame, rstudent, hii, cooksdist;
    t_rstudent::Float64 = 2.5,
    rstudent_faktor::Float64 = 3.0,
    p::Int64 = 2,
    hebel_faktor::Float64 = 3.0,
    t_cooks::Float64 = 4.0,
    ukz_n::Float64 = 1.0,
    ukz_tges::Float64 = 200.0,
    verdacht_min::Int64 = 2
)
    n = nrow(daten)

    # Grenzwerte
    h̄ = hebel_faktor * p / n
    cook_cut = t_cooks / n

    # Einzelflags (fehlende Diagnosewerte gelten als nicht kritisch)
    flag_rstudent = coalesce.(abs.(rstudent) .> t_rstudent, false)
    flag_hebel    = coalesce.(hii .> h̄, false)
    flag_cooks    = coalesce.(cooksdist .> cook_cut, false)
    flag_ukz      = (daten.rMenge .> ukz_n) .&& (daten.tges .< ukz_tges)

    # Hartes RStudent-Kriterium: |RStudent| um den Faktor rstudent_faktor
    # über dem Schwellenwert → alleiniger Ausreißer
    flag_rstudent3 = coalesce.(abs.(rstudent) .> rstudent_faktor * t_rstudent, false)

    # Bitmaske: Bit 0 = RStudent, Bit 1 = Hebelwert, Bit 2 = Cooks, Bit 3 = Ultrakurzzeit,
    # Bit 4 = RStudent extrem
    flags = (Int.(flag_rstudent) .<< 0) .|
            (Int.(flag_hebel)    .<< 1) .|
            (Int.(flag_cooks)    .<< 2) .|
            (Int.(flag_ukz)      .<< 3) .|
            (Int.(flag_rstudent3) .<< 4)

    # Entscheidungsregel — einzige Quelle der Wahrheit ist `ist_ausreißer`
    ist_ausreißer_flagged = ist_ausreißer.(flags)

    # Anzahl erfüllter Diagnose-Flags als Score (0–3), Ultrakurzzeit und das
    # harte RStudent-Kriterium zählen nicht mit, da sie allein schon zum
    # Ausreißer machen
    n_diagnose = Int.(flag_rstudent) .+ Int.(flag_hebel) .+ Int.(flag_cooks)

    # Einstufung: Ausreißer (hard) > Verdacht (Mehrheitsvotum) > OK
    einstufung = [ ifelse(ist_ausreißer_flagged[i], "Ausreißer",
                   ifelse(n_diagnose[i] >= verdacht_min, "Verdacht", "OK"))
                   for i in eachindex(n_diagnose) ]

    return DataFrame(
        :rMenge        => daten.rMenge,
        :tges          => daten.tges,
        :rstudent      => rstudent,
        :hii           => hii,
        :cooksdist     => cooksdist,
        :flag_rstudent => flag_rstudent,
        :flag_hebel    => flag_hebel,
        :flag_cooks    => flag_cooks,
        :flag_ukz      => flag_ukz,
        :flag_rstudent3 => flag_rstudent3,
        :flags         => flags,
        :n_diagnose    => n_diagnose,
        :einstufung    => einstufung,
        :ist_ausreißer => ist_ausreißer_flagged
    )
end

# =========================================================================
# Flags nach der LOO-MAD-Methode: ausgeben (ausreißer_flags_mad)
# und direkt in ein Teil schreiben (ausreißer_flags_setzen!)
# =========================================================================

"""
 #  ausreißer_flags_mad(daten, Reg = :MM, formel = FORMELN[:linear]; kwargs...) -> DataFrame
 #  ausreißer_flags_setzen!(teil::Teil, Reg::Symbol = :MM, formel = FORMELN[:linear]; kwargs...)

 Ausreißer-Flags nach der robusten LOO-MAD-Methode: Pro Beobachtung i wird
 das Verfahren `Reg` ohne i gefittet und daraus `rstudent_mad` (robust
 standardisiertes LOO-Residuum), `hii` (Hebelwert-Diagonale aus dem
 LOO-Vorhersagehebel) und `cooks_mad` (robuste Cook's Distance) berechnet —
 über `loo_mad_diagnostics` mit denselben `fit`-/`predict`-Adaptern,
 Verfahrens-Registry `REG_CORES` und Formel-Katalog `FORMELN` wie in den
 übrigen Validierungsfunktionen aus `Regressionsvalidation.jl`. Fehlende
 Diagnosewerte (MAD-Skala 0) lösen kein Flag aus.

 ## Aufrufe

 - `ausreißer_flags_mad(daten, Reg, formel; kwargs...)` — Diagnose und
   Ausgabe: gibt den Auswertungs-DataFrame (Rohwerte, Flags, Bitmaske
   `flags`, `n_diagnose`, `einstufung`, `ist_ausreißer`) zurück, ohne die
   Daten zu verändern.
 - `ausreißer_flags_setzen!(teil, Reg, formel; kwargs...)` — wie oben auf
   `teil.df` und schreibt zusätzlich das Byte als Spalte `:ausreißer_byte`
   sowie die Einstufung als `:Ausreißer` direkt in `teil.df`.
 - `Reg`: Verfahren aus `keys(REG_CORES)` (`:MM`, `:TheilSen`),
   Standard `:MM`.
 - `formel`: `FormulaTerm` oder Symbol aus `FORMELN` (`:linear`, `:ptfe`),
   Standard `FORMELN[:linear]`.
 - `ausreißer_flags_setzen!(teil, formel; kwargs...)` — Kurzform mit Formel
   statt Verfahren, Reg läuft dann auf `:MM`.

 Die Anzahl der Modellparameter p wird aus der Designmatrix der Formel
 abgeleitet (inkl. Intercept), also p = 2 für `tges ~ rMenge` und damit
 h̄ = p/n. Achtung: `length(formular.rhs)` allein vergäbe den Achsenabschnitt
 und ergäbe h̄ = 1/n — zu streng bzw. falsch.

 Die klassische Diagnose über LinRegOutliers (`createRegressionSetting`,
 `studentizedResiduals`, `hatmatrix`, `cooks`) bleibt über
 `ausreißer_flags(modell, daten; ...)` erreichbar.

 Alle weiteren Keyword-Argumente werden an die Flag-Berechnung
 (`ausreißer_flags`) durchgereicht.

 Achtung: Die Flagspalten werden an `teil.df` angehängt. Regressionen/Filter,
 die alle Spalten von `teil.df` verwenden, sollten zuerst laufen. Die von der
 Formel benötigten Spalten müssen vollständig (ohne `missing`) sein.

 ## Beispiel
 ```julia
 flags_df = ausreißer_flags_mad(teil.df, :MM)     # nur Diagnose + Ausgabe
 ausreißer_flags_setzen!(teil)                   # Standard: :MM, FORMELN[:linear]
 ausreißer_flags_setzen!(teil, :TheilSen, :linear)
 ausreißer_flags_setzen!(teil, FORMELN[:ptfe])   # Kurzform: :MM + PTFE-Kurve
 ```
"""
function ausreißer_flags_mad(daten::AbstractDataFrame, Reg::Symbol = :MM, formel = FORMELN[:linear];
    t_rstudent::Float64 = 2.5,
    rstudent_faktor::Float64 = 3.0,
    hebel_faktor::Float64 = 3.0,
    t_cooks::Float64 = 4.0,
    ukz_n::Float64 = 1.0,
    ukz_tges::Float64 = 200.0,
    verdacht_min::Int64 = 2
)
    # Hilfreicher Hinweis, wenn versehentlich ein Formel-Symbol als Verfahren übergeben wird
    if !haskey(REG_CORES, Reg) && haskey(FORMELN, Reg)
        throw(ArgumentError(
            ":$Reg ist eine Formel aus FORMELN, kein Regressions-Typ. " *
            "Gemeint ist wohl ausreißer_flags_mad(daten, :MM, :$Reg) — " *
            "Verfügbar: " * join(sort(collect(keys(REG_CORES))), ", ")))
    end

    f = _formel_aus_katalog(formel)
    p = size(StatsModels.modelmatrix(StatsModels.ModelFrame(f, daten)), 2)

    diag = loo_mad_diagnostics(daten, Reg, f; auswahl = (:rstudent, :hebel, :cooks))
    return _ausreißer_flags(daten,
        diag.rstudent_mad,
        diag.hii,
        diag.cooks_mad;
        t_rstudent = t_rstudent,
        rstudent_faktor = rstudent_faktor,
        p = p,
        hebel_faktor = hebel_faktor,
        t_cooks = t_cooks,
        ukz_n = ukz_n,
        ukz_tges = ukz_tges,
        verdacht_min = verdacht_min
    )
end

function ausreißer_flags_setzen!(teil::Teil, Reg::Symbol = :MM, formel = FORMELN[:linear];
    t_rstudent::Float64 = 2.5,
    rstudent_faktor::Float64 = 3.0,
    hebel_faktor::Float64 = 3.0,
    t_cooks::Float64 = 4.0,
    ukz_n::Float64 = 1.0,
    ukz_tges::Float64 = 200.0,
    verdacht_min::Int64 = 2
)
    flags_df = ausreißer_flags_mad(teil.df, Reg, formel;
        t_rstudent = t_rstudent,
        rstudent_faktor = rstudent_faktor,
        hebel_faktor = hebel_faktor,
        t_cooks = t_cooks,
        ukz_n = ukz_n,
        ukz_tges = ukz_tges,
        verdacht_min = verdacht_min
    )

    teil.df.ausreißer_byte = flags_df.flags
    teil.df.Ausreißer = flags_df.einstufung
    return flags_df
end

# Kurzform: Formel statt Verfahren -> Reg läuft auf :MM
function ausreißer_flags_setzen!(teil::Teil, formel::StatsModels.FormulaTerm;
    t_rstudent::Float64 = 2.5,
    rstudent_faktor::Float64 = 3.0,
    hebel_faktor::Float64 = 3.0,
    t_cooks::Float64 = 4.0,
    ukz_n::Float64 = 1.0,
    ukz_tges::Float64 = 200.0,
    verdacht_min::Int64 = 2
)
    return ausreißer_flags_setzen!(teil, :MM, formel;
        t_rstudent = t_rstudent,
        rstudent_faktor = rstudent_faktor,
        hebel_faktor = hebel_faktor,
        t_cooks = t_cooks,
        ukz_n = ukz_n,
        ukz_tges = ukz_tges,
        verdacht_min = verdacht_min
    )
end
