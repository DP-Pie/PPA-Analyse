# Hilfsfunktionen zum Erkunden und bearbeiten von DataFrames

"""
 # count_unique
 Zählt die einzigartigen Elemente eines Vectors bzw. Spalte eines DataFramesund gibt diese als Dictonary oder DataFrame zurück. 

 ## Beispiele:
 ```julia
    count_unique(df,:Teil)      # DataFrame mit Spalten-Symbol
    count_unique(df,"Teil")     # DataFrame mit Spalten-String
    count_unique(df,1)          # DataFrame mit Spalten-Zahl
    count_unique([a,b,c],r eturn_df = true) # Vektor mit Dataframeantwort
 ```
"""
function count_unique(df::DataFrame, col::Symbol; return_df::Bool = false)
    return count_unique(df[!, col]; return_df = return_df)
end

function count_unique(df::DataFrame, col::AbstractString; return_df = false)
    return count_unique(df[!, Symbol(col)]; return_df = return_df)
end

function count_unique(df::DataFrame, col::Integer; return_df::Bool = false)
    return count_unique(df[!, col]; return_df = return_df)
end

function count_unique(vec::AbstractVector; return_df::Bool = false)
    c = Dict{eltype(vec), Int64}()
    for t in vec
        c[t] = get(c, t, 0) + 1
    end
    return return_df ? DataFrame((varname = k, Anzahl = v) for (k, v) in c) : c
end

function df_aus_Teiledict(dict::Dict{String,Teil})
    dfs = Vector{DataFrame}(undef,length(keys(dict)))
    for (i,(key,val)) in enumerate(dict)
        dfs[i] = copy(val.df)
        dfs[i].Teil .= val.name
        dfs[i].Gruppe .= val.Gruppe
    end

    return vcat(dfs...)
end

"""
 # f_gruppe_anlegen!

 Funktion zum Anlegen einer Gruppe mit Sammelkontainer für Gruppen mit weniger al `n_min` Einträge

 ## Beispiel

 ```julia
 df_gruppe_anlegen!(df,:Platine; Gruppenname = "Gruppe" , n_min = 15)
 df_gruppe_anlegen!(df,"Aktivität")
 df_gruppe_anlegen!(df,15) 
 ```
"""
function df_gruppe_anlegen!(df::DataFrame, Ziel::Symbol; Gruppenname::String = "Gruppe", Kleinvieh::String = "Sonstiges", n_min::Int64 = 10)
    Anzahl = count_unique(df,Ziel)
    g = Symbol(Gruppenname)

    # df[!,g] .= ""
    # for i in 1:nrow(df)
    #    if Anzahl[df[i,Ziel]] < n_min
    #         df[i,g] = Kleinvieh 
    #    else    
    #        df[i,g] = df[i,Ziel]
    #    end
    # end
    # kürzere Alternative, Aber schwerer zulesen
    df[!, g] = [Anzahl[x] < n_min ? Kleinvieh : x for x in df[!, Ziel]]

    return nothing
end

# Variante mit String als Spaltenname
function df_gruppe_anlegen!(df::DataFrame, Ziel::String; Gruppenname::String = "Gruppe", Kleinvieh::String = "Sonstiges", n_min::Int64 = 10)
    df_gruppe_anlegen!(df,Symbol(Ziel); Gruppenname = Gruppenname, Kleinvieh = Kleinvieh, n_min = n_min)
end

# Variante mit Spaltenindex
function df_gruppe_anlegen!(df::DataFrame, col::Int64; Gruppenname::String = "Gruppe", Kleinvieh::String = "Sonstiges", n_min::Int64 = 10)
    df_gruppe_anlegen!(df,names(df)[col]; Gruppenname = Gruppenname, Kleinvieh = Kleinvieh, n_min = n_min)
end

function collect_diagnistic(teiledict::Dict{String,Teil}, Regression::String)
    results = DataFrame[]

    for (key, teil) in teiledict
        Reg = get(teil.RegressionsModell, Regression, nothing)
        Reg === nothing && continue          # Teil ohne dieses Modell überspringen
        isempty(Reg.Diagnostik) && continue  # diagnostik! noch nicht ausgeführt
        diag_df = copy(Reg.Diagnostik)
        diag_df.Teil .= teil.name
        diag_df.Gruppe .= teil.Gruppe
        push!(results, diag_df)
    end

    return vcat(results...)
end

"""
 # speichere_result_excel(result::DataFrame, pfad::String = "Result.xlsx")

 Schreibt den Result-DataFrame als Excel-Tabelle und färbt die
 :Einschätzung-Spalte je Stufe ein:
 Verwerfen = dunkelrot, Untersuchen = orange, Akzeptabel = gelb,
 Sicher = hellgrün, Sehr sicher = grün. Unbekannte Werte bleiben ungefärbt.
"""
function speichere_result_excel(result::DataFrame, pfad::String = "Result.xlsx")
    farben = Dict(
        "Verwerfen"   => "darkred",     # tiefrot
        "Untersuchen" => "orange",
        "Akzeptabel"  => "yellow",
        "Sicher"      => "lightgreen",  # hellgrün
        "Sehr sicher" => "green"
    )

    XLSX.openxlsx(pfad; mode = "w") do xf
        sheet = XLSX.addsheet!(xf, "Result")
        XLSX.writetable!(sheet, result)

        # Spaltennummer von :Einschätzung (Header Zeile 1, Daten ab Zeile 2)
        col = findfirst(==("Einschätzung"), string.(propertynames(result)))
        col === nothing && throw(ArgumentError("Result hat keine :Einschätzung-Spalte."))

        for (i, stufe) in enumerate(result[!, :Einschätzung])
            farbe = get(farben, String(stufe), nothing)
            farbe === nothing && continue
            XLSX.setFill(sheet, XLSX.CellRef(i + 1, col);
                         pattern = "solid", fgColor = farbe)
        end
    end
    return pfad
end

