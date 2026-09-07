# Kontainer für die Regessionsmodelle und deren Daten
@kwdef mutable struct RegressionsModell
    name::String
    stats::DataFrame #LOSS-Fkt, m, n, std_err, r2, p_value,...
    points::DataFrame #x, y, y_hat, residuals
end

# Kontainer für das Teil mit all seinen Einträgen und den zugehörigen Regressionsdaten
@kwdef mutable struct Teil
    df::DataFrame
    name::String
    TA::Int64
    Zustand::Bool # ja = true, nein = false
    Bezeichnung::String
    Aktivität::String
    n::Int64
    Soll_te::Float64
    Soll_tr::Float64
    RegressionsModell::Dict{String, RegressionsModell}
end

"""
# Funktion zum Gruppieren der Daten nach Teil
Gruppiert die Teile im Datafram nach Teil-Namen und erstellt dafür ein Dictonary mit einem Kontainer-Objekt.
Dies beinhaltet die Daten des Teils, die Anzahl der Einträge, Sollwerte und die zugehörigen Regressionsmodelle.
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

        # Nur die gewünschten Spalten im Teil-DataFrame speichern
        daten = select(
            DataFrame(gruppe),
            [:RMNr, :rMenge, :IstteSek, :Isttrsek, :tges]
        )

        modelle = Dict{String, RegressionsModell}()

        teile[name] = Teil(
            df = daten,
            name = name,
            TA = Int(gruppe.TA[lokal]),
            Zustand = Bool(gruppe.Zustand[lokal]),
            Bezeichnung = String(gruppe.Bezeichnung[lokal]),
            Aktivität = String(gruppe.Aktivität[lokal]),
            n = n,
            Soll_te = Float64(gruppe.Sollte[lokal]),
            Soll_tr = Float64(gruppe.Solltrsek[lokal]),
            RegressionsModell = modelle
        )
    end

    return teile
end

function liste_regressionsmodelle(teil::Teil)
    if isempty(teil.RegressionsModell)
        println("Für Teil '$(teil.name)' sind keine Regressionsmodelle gespeichert.")
        return nothing
    end

    println("Regressionsmodelle für Teil: $(teil.name)")

    for (name, modell) in sort(collect(teil.RegressionsModell))
        print("- ", name, ": ")
        for col in names(modell.stats)[2:1:end]
            print(col, " = ", modell.stats[1, col], ", ")
        end
        println("Punkte: ", nrow(modell.points))
    end

    return nothing
end