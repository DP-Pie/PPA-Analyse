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

function liste_regressionsmodelle(teil::Teil, returnText::Bool = true, returnVec::Bool = false)
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
            h_str = "    $(i): $(key): "
            for col in names(teil.RegressionsModell[key].stats) 
                h_val = truncate_float(teil.RegressionsModell[key].stats[1, col])
                h_str *= "$(col)= $h_val, "
            end
            h_str *= "Messpunkte: $(nrow(teil.RegressionsModell[key].points))"
            println(h_str)
        end
    end

    if returnVec == true
        return Modells
    end

    return nothing
end

