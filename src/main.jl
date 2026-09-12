using Pkg

REQUIRED_PACKAGES = [
    "NativeFileDialog",
    "RobustModels",
    "LinRegOutliers",
    "DataFrames",
    "CSV",
    "GLMakie",
    "JLD2",
    "CairoMakie",
    "Statistics",
    "GLM",
    "StatsModels",
    "StatsAPI"
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
using Statistics
using GLM
using StatsModels
using StatsAPI

include("Konstructs.jl")

df_path = pick_file(filterlist="*csv")
#df_path = "../PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV"
df = CSV.read(df_path, DataFrame)

Teiledict = erstelle_teile(df)

# Diagramme anzeigen
include("Diagrams.jl")
# Anzahl der Einträge je Teil, absteigend sortiert
GLMakie.activate!(inline=false)
show_einträge()

# Speichern der Eintragsverteilung als SVG-Datei
CairoMakie.activate!()
save("eintragsverteilung.svg", fig)
vscodedisplay(df)

include("Regressionen.jl")



# Theil-Sen Regression für jedes Teil durchführen, das mindestens 20 Einträge hat
for (key,Teil) in Teiledict
    if Teil.n >= 95
        #println("$key hat $(Teil.n) Einträge.")
        # Regressionen durchführen
        theil_sen_regression!(Teil)
        mm_regression!(Teil)
        # Regressionsdaten ausgeben, nur Text
        liste_regressionsmodelle(Teil, true)
    end
end 

using Random

# --><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# todo 
function bootstrap(teil::Teil, foo::Function, B::Int64=1_000)
    
        daten = dropmissing(select(teil.df, [:rMenge, :tges]),[:rMenge, :tges])

        if nrwo(daten) <= 20
            throw(ArgumentError("Teil: $(teil.name) hat weniger als 20 Datenpunkte, Für die Bootstrap-Methode ungeeignet"))
            return nothing
        end

        x = Float64.(daten.rMenge)
        y = Float64.(daten.tges)

        xlen = length(x)

        idx = rand(1:xlen,xlen)

        m_boot = Vector{Float64}(undef,B)
        n_boot = Vector{Float64}(undef,B)

        xrnd = x[idx]
        yrnd = y[idx]

        for i in 1:1:B    
            myReg = foo(xrnd,yrnd)
            m_boot[i] = myReg.stats.m[1]
            n_boot[i] = myReg.stats.n[1]
        end





end

# QDK...Quartilsdispersionskoeffizient 
function QDK(Vec::Vector{Float64},MedianForm::Bool=ture)
    Q1, Median, Q3 = quantile(Vec,[0.25,0.5,0.75])

    if MedianForm == true
        if Median == 0.0 
            throw(DivisionByZero("Median der Daten ist Null. Berechnung nicht möglich."))
        end
        result = (Q3 - Q1) / Median
    else
        ∑Q = Q3 + Q3
        if ∑Q == 0.0
            throw(DivisionByZero("Summer der Datenquantile 3 und 1 ist Null. Berechnung nicht möglich."))          
        end
    end
    return result
end

Teildictselection_u20 = Dict{String,Teil}()
for (key,teil) in Teiledict
    if teil.n < 20
       Teildictselection_u20[teil.name] = teil
    end
end

@save "Teiledict.jld2" Teiledict
@save "Teildictselection_u20.jld2" Teildictselection_u20

#Datenframe aus Selektion erstellen.
teile_df = DataFrame(
    Aktivität = [teil.Aktivität for teil in values(Teiledict)],
    Teil = [teil.name for teil in values(Teiledict)],
    n = [teil.n for teil in values(Teiledict)],
    Bezeichnung = [teil.Bezeichnung for teil in values(Teiledict)]
)
#Datenframe sortieren 
sort!( teile_df, [:Aktivität, :n, :Teil], rev = [false, true, false])

insertcols!( teile_df, 1, :ID => 1:nrow(teile_df))

#leere Spalte für die Gruppierung
insertcols!( teile_df, :Gruppe => Vector{Union{Missing, String}}(missing, nrow(teile_df)))

function gruppe_setzen!(
    df::DataFrame,
    von::Int,
    bis::Int,
    gruppe::String
)
    df[
        (df.ID .>= von) .& (df.ID .<= bis),
        :Gruppe
    ] .= gruppe

    return df
end
@load "Teildictselection_u20.jld2" Teildictselection_u20
keys(Teildictselection_u20)
teile_df = CSV.read("PPA-BDE_Aktivitäten.csv", DataFrame)
vscodedisplay(teile_df)

length(unique(teile_df.Aktivität)) #249
verpackung_df = filter(:Aktivität => ==("Verpacken"), teile_df)
vscodedisplay(verpackung_df)

gruppe_setzen!(teile_df, 986, 992, "Modular-Kabel")


### prefixselection richtig einpflegen
function gruppe_aus_teil(teilname; usePrefix::String)
    if ismissing(teilname)
        return missing
    end

    teilname = string(teilname)

    # teilnamee ignorieren, die mit zwei Buchstaben beginnen
    if !startswith(teilname,usePrefix)
        return missing
    end

    # Abschließendes T entfernen
    teilname = replace(teilname, r"T$" => "")

    # Ziffern am Ende suchen
    match_result = match(r"(\d+)$", teilname)

    # Keine Endziffern vorhanden
    isnothing(match_result) && return missing

    ziffern = match_result.captures[1]

    # Die letzten vier Ziffern verwenden und links mit Nullen auffüllen
    return lpad(last(ziffern, min(4, length(ziffern))), 4, '0')
end

function gruppe_aus_teil(teilname)
    if ismissing(teilname)
        return missing
    end

    teilname = string(teilname)

    # teilnamee ignorieren, die mit zwei Buchstaben beginnen
    if occursin(r"^[A-Za-z]{2}", teilname)
        return missing
    end

    # Abschließendes T entfernen
    teilname = replace(teilname, r"T$" => "")

    # Ziffern am Ende suchen
    match_result = match(r"(\d+)$", teilname)

    # Keine Endziffern vorhanden
    isnothing(match_result) && return missing

    ziffern = match_result.captures[1]

    # Die letzten vier Ziffern verwenden und links mit Nullen auffüllen
    return lpad(last(ziffern, min(4, length(ziffern))), 4, '0')
end

for i in 1:nrow(teile_df)
    if isequal(teile_df[i, :Aktivität], "Verpacken") &&
       ismissing(teile_df[i, :Gruppe])

        teile_df[i, :Gruppe] = gruppe_aus_teil(teile_df[i, :Teil])
    end
end

vscodedisplay(teile_df)

CSV.write("PPA-BDE_Aktivitäten2.csv", teile_df)