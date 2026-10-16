# Kontainer für die Regessionsmodelle und deren Daten

"""
 # RegressionsModell

 Kontainer für Regressionsmodelle.

 ## Strucktur

    RegressionsModell
    ├── name::String        ...Names des Regressionsmodells
    ├── modell              ...Regressionstyp aus der Regressionsfunktion
    ├── stats::DataFrame    ...Dataframe mit Modelldaten (Anstieg, Achsenabschnitt, Fehler...)
    ├── points::DataFrame   ...Dataframke der verwendeten und bearbeiten Punktdaten 
    └── Diagnostik::DataFrame ...Bootstrap- und LOO-Kennzahlen je Koeffizient
                                 (leer bis `diagnostik!` ausgeführt wurde)
"""
@kwdef mutable struct RegressionsModell
    name::String
    modell
    stats::DataFrame    #LOSS-Fkt, m, n, std_err, r2, p_value,...
    points::DataFrame   #x, y, y_hat, residuals
    # Bootstrap- und LOO-Kennzahlen je Koeffizient; Standard leer,
    # gefüllt von diagnostik! (Regressionsvalidation.jl). Neues Feld mit
    # Default, damit bestehende Konstruktor-Aufrufe unverändert laufen.
    Diagnostik::DataFrame = DataFrame()
end

"""
 # Teil

 Kontainer für ein Bauteil mit allen Einträgen und zugehörigen Regressionsdaten.
 ## Strucktur

    Teil
    ├── df::DataFrame           ...gesammelte Rohdaten für das Bauteil
    ├── name::String            ...Bauteilbezeichnung
    ├── TA::Int64               ...Teileart
    ├── Zustand::Bool           ...Aktives Bauteil Ja/Nein
    ├── Bezeichnung::String     ...Beschreibung des Bauteils
    ├── Aktivität::String       ...Grobegruppierung der Aktivität/Bararbeitung des Bauteils
    ├── Gruppe::String          ...Gruppierung für die Regression
    ├── n::Int64                ...Anzahl der RMNr-Einträge
    ├── Soll_te::Float64        ...geplante Solzeit für die Bauteilproduktionszeit (n=1)
    ├── Soll_tr::Float64        ...geplante Sollzeit für die Rüstzeit
    └── RegressionsModell::Dict{String, RegressionsModell}
"""
# Kontainer für das Teil mit all seinen Einträgen und den zugehörigen Regressionsdaten
@kwdef mutable struct Teil
    df::DataFrame
    name::String
    TA::Int64
    Zustand::Bool # ja = true, nein = false
    Bezeichnung::String
    Aktivität::String
    Gruppe::String
    n::Int64
    Soll_te::Float64
    Soll_tr::Float64
    RegressionsModell::Dict{String, RegressionsModell}
end

"""
 # TEIL_METADATA

 Blockliste der Metadaten-Spalten, die NICHT in `teil.df` übernommen werden.
 Alle übrigen Spalten der Rohdaten fließen automatisch in `teil.df` ein, neue
 Messspalten (z.B. :Länge, :RM, :BeinLänge) müssen also nirgends ergänzt werden.
 Kommen später neue Metadaten-Spalten hinzu, sind sie hier zu ergänzen.
"""
const TEIL_METADATA = [
    :Teil, :TA, :Bezeichnung, :Aktivität, :AktivitätNeu,
    :Gruppe, :GruppeNeu, :Zustand, :Sollte, :Solltrsek, :Linkname
]

"""
 # erstelle_teile(df::DataFrame)

 Erstellt aus einem DataFrame ein Dictionary von `Teil`-Objekten.
 Gruppiert die Daten nach der Spalte `:Teil`, extrahiert Metadaten wie TA, Zustand, Bezeichnung, etc....
 aus der Zeile mit der höchsten RMNr und speichert die Rohdaten im `df`-Feld jedes Teils.

 ## Warum die höchste RMNr verwendet wird?

 Weil dies wichtig ist für die Festlegung von Soll_te (Soll-Bauteilzeit) und Soll_tr (Soll-Rüstzeit), da diese sich im laufe der Zeit ändern können. 
"""
function erstelle_teile(df::DataFrame)
    teile = Dict{String, Teil}()

    # Teil = missing wird nicht als eigenes Teil verarbeitet
    df_ohne_missing = dropmissing(df, :Teil)

    for gruppe in groupby(df_ohne_missing, :Teil)
        name = string(gruppe.Teil[1])
        n = nrow(gruppe)

        # Zeile mit der größten gültigen RMNr bestimmen
        gueltig = findall(!ismissing, gruppe.RMNr)

        isempty(gueltig) &&
            throw(ArgumentError("Teil '$name' enthält keine gültige RMNr."))

        lokal = gueltig[argmax(gruppe.RMNr[gueltig])]

        # Alle Messspalten übernehmen (Blockliste statt fester Auswahlliste):
        # intersect, damit in anderen CSV-Varianten fehlende Metadaten-Spalten
        # nicht zu einem Fehler führen
        daten = select(
            DataFrame(gruppe),
            Not(intersect(TEIL_METADATA, Symbol.(names(gruppe))))
        )

        modelle = Dict{String, RegressionsModell}()

        teile[name] = Teil(
            df = daten,
            name = name,
            TA = Int(gruppe.TA[lokal]),
            Zustand = Bool(gruppe.Zustand[lokal]),
            Bezeichnung = String(gruppe.Bezeichnung[lokal]),
            Aktivität = String(gruppe.AktivitätNeu[lokal]),
            Gruppe = String(gruppe.Gruppe[lokal]),
            n = n,
            Soll_te = Float64(gruppe.Sollte[lokal]/n),
            Soll_tr = Float64(gruppe.Solltrsek[lokal]),
            RegressionsModell = modelle
        )
    end

    return teile
end

"""
 # erstelle_teile_aus_gruppe(gdf::GroupedDataFrame) -> Dict{String, Teil}

 Erstellt aus einem GroupedDataFrame (z.B. AOI-Platinen nach :Gruppe) ein
 Dictionary von `Teil`-Objekten — ein Teil je Gruppe. Im Gegensatz zu
 `erstelle_teile` werden keine Metadaten-Spalten (TA, Zustand, Sollte, ...)
 benötigt, da Datenquellen wie BDE02 sie nicht liefern; die Felder werden
 mit Platzhaltern belegt:
 TA = 0, Zustand = true, Bezeichnung = "", Aktivität = "",
 Soll_te = Soll_tr = NaN.

 ## Voraussetzungen an die Gruppen-Daten

 - Gruppierung nach genau EINER Spalte (deren Wert wird Teilname und :Gruppe).
 - Die Messspalten der Regression (:tges, :rMenge) müssen vorhanden und
   ohne missing sein (ggf. vorher `dropmissing!`).

 ## Beispiel
 ```julia
 AOI_teile = erstelle_teile_aus_gruppe(groupby(AOI_df, :Gruppe))
 ```
"""
function erstelle_teile_aus_gruppe(gdf::GroupedDataFrame)
    gruppen_spalten = DataFrames.groupcols(gdf)
    length(gruppen_spalten) == 1 ||
        throw(ArgumentError("erstelle_teile_aus_gruppe unterstützt nur genau eine Gruppierungsspalte, erhalten: $(gruppen_spalten)"))
    gruppen_spalte = only(gruppen_spalten)

    teile = Dict{String, Teil}()

    for gruppe in gdf
        name = string(gruppe[1, gruppen_spalte])
        daten = select(gruppe, Not(gruppen_spalte))

        teile[name] = Teil(
            df = daten,
            name = name,
            TA = 0,
            Zustand = true,
            Bezeichnung = "",
            Aktivität = "",
            Gruppe = name,
            n = nrow(gruppe),
            Soll_te = NaN,
            Soll_tr = NaN,
            RegressionsModell = Dict{String, RegressionsModell}()
        )
    end

    return teile
end

function erstelle_teile_aus_gruppe(df::DataFrame,Gruppe::Symbol)
    return erstelle_teile_aus_gruppe(groupby(df,Gruppe))
end

function erstelle_teile_aus_gruppe(df::DataFrame,Gruppe::String)
    return erstelle_teile_aus_gruppe(groupby(df,Symbol(Gruppe)))
end

"""
 # spalte_gruppe_in_teile(teil::Teil; meta = nothing) -> Dict{String, Teil}
 # spalte_gruppe_in_teile(gruppen::Dict{String, Teil}; meta = nothing) -> Dict{String, Teil}

 Zerlegt Gruppen-`Teil`-Objekte (aus `erstelle_teile_aus_gruppe`) in ihre
 Einzelteile. Aufgespalten wird über die `:Teil`-Spalte von `teil.df`; die
 Regressionsdaten der Gruppe werden als Referenz an jedes Einzelteil
 übergeben. Ein Modell, viele Zeiger: Speicher und Konsistenz bleiben
 erhalten, ein `diagnostik!` über ein Einzelteil aktualisiert damit das
 Gruppenmodell für alle zugehörigen Einzelteile (gleicher Datenstand,
 gewollt). Wer isolierte Kopien braucht, übergibt stattdessen
 `deepcopy(teil.RegressionsModell)`.

 ## Metadaten über `meta` nachziehen

 BDE02 liefert keine Metadaten (TA, Zustand, Bezeichnung, AktivitätNeu,
 Sollte, Solltrsek), auf die `erstelle_teile` zugreift — `erstelle_teile`
 selbst würde also fehlschlagen. Mit `meta = Teiledict` (BDE01-Sammlung aus
 `erstelle_teile`, gleiche Teilnamen) werden stattdessen TA, Zustand,
 Bezeichnung, Aktivität, Soll_te und Soll_tr aus dem gleichnamigen Eintrag
 übernommen. Ohne Treffer — oder ohne `meta` — bleiben die Platzhalter wie
 bei `erstelle_teile_aus_gruppe` (TA = 0, Zustand = true, Bezeichnung = "",
 Aktivität = "", Soll_te = Soll_tr = NaN); `Gruppe` erhält in jedem Fall
 den Namen der Herkunftsgruppe, sodass die Provenienz erhalten bleibt.
 Stand 2026-10: Alle 30 AOI-Teile aus BDE02 sind im Teiledict vertreten.

 Die Kennzahlen im `RegressionsModell` (stats, points, Diagnostik) bleiben
 Gruppeneigenschaften: `points` enthält weiterhin alle Punkte der Gruppe,
 nicht nur die des Einzelteils.

 ## Beispiel
 ```julia
 AOI_testdict = spalte_gruppe_in_teile(AOI_Teile_group; meta = Teiledict)
 AOI_testdict["OI-VAG01e-01"].RegressionsModell["Theil-Sen"].stats
 ```
"""
function spalte_gruppe_in_teile(teil::Teil; meta::Union{Nothing, Dict{String, Teil}} = nothing)
    :Teil in propertynames(teil.df) ||
        throw(ArgumentError(
            "spalte_gruppe_in_teile braucht eine :Teil-Spalte in teil.df " *
            "(Gruppen aus erstelle_teile_aus_gruppe haben eine, " *
            "Teile aus erstelle_teile nicht)."))

    teile = Dict{String, Teil}()

    for gruppe in groupby(teil.df, :Teil)
        name = String(gruppe.Teil[1])
        daten = select(gruppe, Not(:Teil))

        # Metadaten aus dem Lookup übernehmen, sonst Platzhalter
        quelle = meta === nothing ? nothing : get(meta, name, nothing)

        teile[name] = Teil(
            df = daten,
            name = name,
            TA = quelle === nothing ? 0 : quelle.TA,
            Zustand = quelle === nothing ? true : quelle.Zustand,
            Bezeichnung = quelle === nothing ? "" : quelle.Bezeichnung,
            Aktivität = quelle === nothing ? "" : quelle.Aktivität,
            Gruppe = teil.name,
            n = nrow(gruppe),
            Soll_te = quelle === nothing ? NaN : quelle.Soll_te,
            Soll_tr = quelle === nothing ? NaN : quelle.Soll_tr,
            RegressionsModell = teil.RegressionsModell
        )
    end

    return teile
end

"""
 # spalte_gruppe_in_teile(gruppen::Dict{String, Teil}; meta = nothing) -> Dict{String, Teil}

 Fasst die Einzelteile aller Gruppen zu einem Dictionary zusammen. Kommt ein
 Teilname in mehreren Gruppen vor (sollte bei Platine-Kodierung im Namen
 nicht passieren), wird ein Fehler geworfen statt still zu überschreiben.
 `meta` (z.B. das BDE01-Teiledict) wird an jeden Einzelsplit durchgereicht.
"""
function spalte_gruppe_in_teile(gruppen::Dict{String, Teil}; meta::Union{Nothing, Dict{String, Teil}} = nothing)
    teile = Dict{String, Teil}()

    for (gruppenname, gruppe) in gruppen
        for (name, teil) in spalte_gruppe_in_teile(gruppe; meta = meta)
            haskey(teile, name) && throw(ArgumentError(
                "Teil '$name' kommt in mehr als einer Gruppe vor " *
                "(zuletzt in '$gruppenname') — Zusammenfassen würde " *
                "still Einträge überschreiben."))
            teile[name] = teil
        end
    end

    return teile
end

"""
# truncate_float(x; digits=3)

Kürzt die Dezimaldarstellung einer Zahl auf `digits` Nachkommastellen.
Falls die Zahl mehr Dezimalstellen hat, werden diese mit `...` abgekürzt.
"""
function truncate_float(x; digits=3)
    s = string(x)
    if occursin('.', s)
        int_part, dec_part = split(s, '.')
        if length(dec_part) > digits
            return "$int_part.$(dec_part[1:digits])..."
        end
    end
    return s
end

"""
 # liste_regressionsmodelle(teil::Teil; returnText::Bool=true, returnVec::Bool=false)

 Listet die Regressionsmodelle eines Teils auf.
 Als Default wird nur ein Text ausgegeben (returnText::Bool=true).

 ## Argumente
 - `returnText`: Gibt die Modelle als formatierten Text in der Konsole aus.
 - `returnVec`: Gibt die Modellnamen als `Vector{String}` zurück.

 Falls keine Modelle vorhanden sind, wird eine Meldung ausgegeben (falls `returnText=true`).
"""
function liste_regressionsmodelle(teil::Teil; returnText::Bool = true, returnVec::Bool = false)
    if isempty(teil.RegressionsModell)
        if returnText == true
            println("Für Teil '$(teil.name)' sind keine Regressionsmodelle gespeichert.")
        end
        return nothing
    end

    Keynames = sort(collect(keys(teil.RegressionsModell))) # key gibt ein dict zurück, collect konvertiert es in einen Vektor
    Modells = Vector{String}(undef, length(Keynames))

    if returnText == true
        println("Regressionsmodelle für Teil: $(teil.name)")
    end

    for (i, key) in enumerate(Keynames)
        Modells[i] = key
        if returnText == true
            stats = teil.RegressionsModell[key].stats
            # stats kann mehrere Zeilen haben (MM: eine je Term/Koeffizient)
            for r in 1:nrow(stats)
                h_str = "    $(i): $(key): "
                for col in names(stats)
                    h_val = truncate_float(stats[r, col])
                    h_str *= "$(col)= $h_val, "
                end
                if r == nrow(stats)
                    h_str *= "Messpunkte: $(nrow(teil.RegressionsModell[key].points))"
                end
                println(h_str)
            end
        end
    end

    if returnVec == true
        return Modells
    end

    return nothing
end

# --- Konstanten und Structs für die Monte-Carlo-Simulation
# Workaround für die Doppeldeklaration in Pluto.jl bei neuladen der Dateien.
# Konstante hat einen Festenspeicher, aber die Felder sind änderbar.
Base.@kwdef mutable struct MC_Konstanten
    # --- Arbeitszeiten ---
    Arbeit_start::Float64 = 0.0
    Arbeit_ende::Float64 =  9.25 * 60 * 60
    # --- Abfertigung, die "das-machen-wir-noch-Fertigzeit" ---
    Abfertigung::Float64 = 30.0 * 60
    # --- Pausen ---
    Frühstückspause_start::Float64 = 2.5 * 60 * 60
    Frühstückspause_ende::Float64 = 2.75 * 60 * 60
    Frühstückspause::Float64 = 15.0 * 60
    Mittagspause_start::Float64 = 5.75 * 60 * 60
    Mittagspause_ende::Float64 = 6.25 * 60 * 60
    Mittagspause::Float64 = 30.0 * 60
    # --- Produktionsparameter ---
    Rüstzeit::Float64 = 10.0 * 60               # tr
    Bauteilzeit::Float64 = 3.0 * 60             # te
    Rüstzeit_Streuung::Float64 = 0.3
    Bauteilzeit_Streuung::Float64 = 0.3
    # --- Bauteilanzahl ---
    Bauteile_min::Int64 = 2
    Bauteile_max::Int64 = 100
end

function Base.show(io::IO, c::MC_Konstanten)
    println(io, "MC_Konstanten (Simulationsparameter)")
    println(io, repeat("=", 40))

    # Arbeitszeiten
    println(io, "\nArbeitszeit:")
    println(io, "   Start: ", round(c.Arbeit_start / 3600; digits=2), " h")
    println(io, "   Ende:  ", round(c.Arbeit_ende / 3600; digits=2), " h (", c.Arbeit_ende, " s)")

    # Pausen
    println(io, "\nPausen:")
    println(io, "   Frühstück: ", round(c.Frühstückspause_start / 3600; digits=2), " h - ",
            round(c.Frühstückspause_ende / 3600; digits=2), " h (",
            round(Int, c.Frühstückspause / 60), " min)")
    println(io, "   Mittag:    ", round(c.Mittagspause_start / 3600; digits=2), " h - ",
            round(c.Mittagspause_ende / 3600; digits=2), " h (",
            round(Int, c.Mittagspause / 60), " min)")

    # Produktionsparameter
    println(io, "\nProduktionszeiten:")
    println(io, "   Rüstzeit:       ", round(Int, c.Rüstzeit / 60), " min (+/-",
            round(Int, c.Rüstzeit_Streuung * 100), "%)")
    println(io, "   Bauteilzeit:    ", round(Int, c.Bauteilzeit / 60), " min (+/-",
            round(Int, c.Bauteilzeit_Streuung * 100), "%)")

    # Bauteilanzahl
    println(io, "\nBauteilanzahl:")
    println(io, "   Min: ", c.Bauteile_min, " | Max: ", c.Bauteile_max)

    # Abfertigung
    println(io, "\nAbfertigung:")
    println(io, "   ", round(Int, c.Abfertigung / 60), " min")
end

const MCconst = MC_Konstanten()

# --- Job-Kontainer ---
@kwdef mutable struct Job  # Großbuchstabe (Konvention für Typen)
    ID :: Int64
    Start :: Float64
    Ende :: Float64
    Bauteile :: Int64
    Bauteilzeit :: Float64  # Hier: Bauteilzeit pro Stück (nicht Gesamtzeit!)
    Rüstzeit :: Float64
    Restzeit :: Float64
    Pause :: Float64
    Gesamtzeit :: Float64
end

# --- Schöner Darstellung bei der Abfrage im REPL ---
function Base.show(io::IO, j::Job)
    println("Job-ID: $(j.ID)\nn: $(j.Bauteile)\nRüstzeit: $(j.Rüstzeit)\nBauteilzeit: $(j.Bauteilzeit)\nStart: $(j.Start)\nEnde: $(j.Ende)\nRestzeit: $(j.Restzeit)\nPause: $(j.Pause)\nGesamtzeit: $(j.Gesamtzeit)")
end

# =========================================================================
# Modellkontainer Theil-Sen
# =========================================================================
"""
 # TheilSenModell

 Leichtgewichtiges Modellobjekt der Theil-Sen-Regression. Die MM-Regression
 speichert mit `RobustModels.RobustLinearModel` ein fertiges Modell aus dem
 Paket; für Theil-Sen gibt es kein solches Pendant, deshalb hält dieser
 Struct die Informationen des Fits zusammen:

    TheilSenModell
    ├── formel::StatsModels.FormulaTerm  ...verwendete Regressionsformel
    ├── coef::Vector{Float64}            ...Koeffizienten, Intercept zuerst (wie bei StatsModels)
    ├── yvar::Symbol                     ...Spaltenname der Zielgröße
    └── xvar::Symbol                     ...Spaltenname des Prädiktors

 Implementiert `StatsAPI.coef` und `StatsAPI.predict`, damit es sich im
 weiteren Code wie ein "echtes" Regressionsmodell behandeln lässt.
"""
@kwdef struct TheilSenModell
    formel::StatsModels.FormulaTerm
    coef::Vector{Float64}   # Intercept zuerst, wie bei StatsModels/StatsAPI
    yvar::Symbol
    xvar::Symbol
end

"""
 # TheilSenModell(formel, daten) -> TheilSenModell

 Fit-Konstruktor: passt Theil-Sen über `Theil_Sen_Coefs` an `daten` an und
 speichert Formel, Koeffizienten und Spaltennamen. Erlaubt sind nur Formeln
 der Art `y ~ x` (Prüfung wie in `Theil_Sen_Coefs`).
"""
function TheilSenModell(formel::StatsModels.FormulaTerm, daten::AbstractDataFrame)
    koeffizienten = Theil_Sen_Coefs(formel, DataFrame(daten))
    yvar = only(collect(StatsModels.termvars(formel.lhs)))
    xvar = only(collect(StatsModels.termvars(formel.rhs)))
    return TheilSenModell(formel, koeffizienten, yvar, xvar)
end

# Schnittstelle wie bei einem Modell aus StatsModels/RobustModels
StatsAPI.coef(m::TheilSenModell) = m.coef
StatsAPI.predict(m::TheilSenModell, daten::AbstractDataFrame) =
    m.coef[1] .+ m.coef[2] .* Float64.(daten[!, m.xvar])
StatsAPI.predict(m::TheilSenModell, x::AbstractVector) =
    m.coef[1] .+ m.coef[2] .* Float64.(x)

