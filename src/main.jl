using Pkg

REQUIRED_PACKAGES = [
    "NativeFileDialog",
    "RobustModels",
    "LinRegOutliers",
    "DataFrames",
    "CSV",
    "GLMakie",
    "JLD2",
    "CairoMakie"
]

# Installiere fehlende Pakete
for package in REQUIRED_PACKAGES
    Base.find_package(package) === nothing && Pkg.add(package)
end

using NativeFileDialog
using RobustModels
using LinRegOutliers
using DataFrames
using CSV
using GLMakie
using JLD2
using CairoMakie
df_path = pick_file(filterlist="*csv")
df = CSV.read(df_path, DataFrame)

# Float-Werte korrekt anzeigen druch runden.
transform!(df, :tges => ByRow(x -> round(x, digits=1)) => :tges)
transform!(df, :Isttrsek => ByRow(x -> round(x, digits=1)) => :Isttrsek)

# :Zustand von Nein/Ja in Boolean umwandeln.
transform!(
    df,
    :Zustand => ByRow(x ->
        ismissing(x) ? missing :
        lowercase(strip(string(x))) == "ja"
    ) => :Zustand
)

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

Teiledict = erstelle_teile(df)

sortierte_teile = sort(
    collect(values(Teiledict)),
    by = teil -> teil.n,
    rev = true
)

anzahlen = [teil.n for teil in sortierte_teile]
x = 1:length(anzahlen)

farben = cgrad(
    :viridis,
    length(anzahlen),
    categorical = true
)

fig = Figure(size = (1000, 600))
y_max = maximum(anzahlen) +10
ax = Axis(
    fig[1, 1],
    xlabel = "Teile, absteigend sortiert",
    ylabel = "Anzahl der Einträge",
    title = "Eintragsverteilung der Teile",

    limits = (-10, 2600, 0, y_max),
    xticks = 0:100:2600,
    yticks = 0:10:y_max,

    backgroundcolor = :white
)

barplot!(
    ax,
    x,
    anzahlen;
    colorrange = (1, max(length(x), 2)),
    strokewidth = 0
)

gesamt = sum(anzahlen)
maximalwert = maximum(anzahlen)
anzahl_teile = length(anzahlen)
prozent_bis_30 = 100 * count(anzahlen .<= 30) / anzahl_teile
prozent_bis_20 = 100 * count(anzahlen .<= 20) / anzahl_teile
prozent_bis_10 = 100 * count(anzahlen .<= 10) / anzahl_teile

text!(
    ax,
    2550,
    y_max * 0.95,
    text = """
Rückmeldungen gesamt: $gesamt
EInträge je Teil: $anzahl_teile
Maximalwert: $maximalwert

Teile ≤ 30 Einträge: $(round(prozent_bis_30, digits=1)) %
Teile ≤ 20 Einträge: $(round(prozent_bis_20, digits=1)) %
Teile ≤ 10 Einträge: $(round(prozent_bis_10, digits=1)) %
""",
    align = (:right, :top),
    fontsize = 16,
    color = :black,
    space = :data
)
fig

CairoMakie.activate!()
save("eintragsverteilung.svg", fig)
vscodedisplay(df)

@save "Teiledict.jld2" Teiledict
@load "Teiledict.jld2" Teiledict