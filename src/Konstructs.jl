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

