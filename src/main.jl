"""
# Präambel 
"""
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
    "StatsAPI",
    "MixedModels",
    "CategoricalArrays",
    "Random",
    "Pluto"
] 
# "https://github.com/patrickm663/YeoJohnson.jl" ersetzte durch eigenen Funktion durch Hin- und Rücktransformation. 
# using YeoJohnson            # Datentransformation für Heteroskedastizität (Variablenabhängige Streuung)

# Installiere fehlende Pakete
for package in REQUIRED_PACKAGES
    Base.find_package(package) === nothing && Pkg.add(package)
end

using NativeFileDialog      # komfortable Dateien öffnen und Speichern
using RobustModels          # Regressionssammlung für robuste Statistik
using LinRegOutliers        # Ausreißer Identifizierung
using DataFrames            # Datenhandling
using CSV                   # CSV-Datein-Handling
using CairoMakie            # Plot Aufbereitung für Veröffentlichungen/PDFs/... 
using GLMakie               # Interaktive Plots
GLMakie.activate!(inline=false) # Bild in VSCodeConsole deaktivieren
using JLD2                  # Speichern/Laden von eignen Structs und Datenstruckturen
using Statistics            # Die wichtigsten statistischen Formeln
#using GLM                   # Backbone für die tiefergehenede statistische Analysen und Regressionen 
using StatsModels           # Bereitstehhen der "gewöhnlichen" Abstacktionen und Hilfsfunktionen zum spezifizieren, annähren und evaluieren von Modellen. 
using StatsAPI              # Bündelt Statistische funktionen und soll Namespace-Konflikte vermeiden
using MixedModels           # Lineare gemischte Modelle
using CategoricalArrays     # Kategoriale Gruppierungsvariablen
using Random                # Funktionserweiterung für den Zufallsgenerator
using LinearAlgebra         # Matrixoperationen und anders mathematisches gedöns
#using Pluto
using Dates


include("Konstructs.jl")                # Kontainervariablen und deren Hilfsfunktionen
include("Diagrams.jl")                  # Funktionen zur Datendarstellung
include("TideyUpData.jl")               # Hilfsfunktionen zum bearbeiten von Datensätzen
include("YeoJohnson_Transformation.jl") # Yeo-Johnson-Transformation: λ-Schätzung, Vorwärts/Rücktransformation
include("Regressionsmethoden.jl")       # Funktionen für die Regressionenmethoden wie Theil Sen, MM, ...
include("Regressionsvalidation.jl")     # Funktion zum Validieren des Regressionsmodells
include("Ausreißer_Flags.jl")           # Flag-System zur Ausreißerklassifikation
include("Montecarlo_Sim.jl")            # Montecarlo-Simulation zum Abschätzen des Pausenzeitfehlers
include("whittaker_eiler_smoother.jl")  # Smoother/Filter ohne Verzögerung im Output


"""
----
# Erstellen eines Sammlung aller Teil mit deren Einträge
----
Ausagngslage ist die CSV-Datei mit den RMNr. ab 43_000.
Diese wurde vorher bereinigt. Das Vorgehen ist in "DF_Clean_Up.jl" zu lesen.
Spätere Bearbeitungen für die Korrekturen der Aktivitäten und Gruppierungen werden leider lückenhaft und unsauber bzw. schlechter nachvollziehbar.
Eine Volle Rekonstruktion des Vorgehens ist somit nicht möglich.
Es bleibt die Arbeit mit dem Endergebnis. 

"""
#df_path = pick_file(filterlist="*csv")
df_path = "PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV"
df = CSV.read(df_path, DataFrame)
vscodedispaly(df)

Teiledict = erstelle_teile(df)
@save "Teiledict.jld2" Teiledict
@load "Teiledict.jld2" Teiledict

# Speichern der Eintragsverteilung als SVG-Datei
CairoMakie.activate!()
fig = show_einträge(Teiledict)
save("eintragsverteilung.svg", fig)

# Theil-Sen Regression und MM-Regression für jedes Teil durchführen, das mindestens 95 Einträge hat
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

# PTFE-Schlauch-Gruppe erstellen
for i in 1:nrow(teile_df)
    if isequal(teile_df[i, :Aktivität], "Schlauch ablängen") &&
       isequal(teile_df[i, :Gruppe], "missing")

        teile_df[i, :Gruppe] = Gruppe_mit_Länge_erstellen(teile_df[i, :Bezeichnung],"PTFE","Länge=","mm")
    end
end

for i in 1:nrow(df)
    if isequal(df[i, :Aktivität], "Schlauch ablängen") &&
       isequal(df[i, :Gruppe], "missing")

        df[i, :Gruppe] = Gruppe_mit_Länge_erstellen(df[i, :Bezeichnung],"PTFE","Länge=","mm")
    end
end
vscodedisplay(df)
Teiledict = erstelle_teile(df)
CSV.write("PPA-BDE01_ab RMNr43000_Daniels_Spielwiese_20260903_clean.CSV", df)

# versuch einer Multiplen-Regression
PTFEdict = Dict{String,Teil}()
for (key,teil) in Teiledict # loop über ein Dictonary wird mit key-wert-Paaren bewerkstelligt, alternative For entry in value(Dict)
    if startswith(teil.Gruppe,"PTFE")
        PTFEdict[key] = teil
    end
end

c = 0
for teil in values(PTFEdict)
    c += teil.n
    println("$(teil.name): $(teil.n)")
end
c

# Datensatz der PTFE-Schlauche mit Länge anlegen 
PTFE_DF = DataFrame()  # Leerer DataFrame mit den Spalten von `teil.df`
for (key, teil) in PTFEdict
    h_df = copy(teil.df)  # Kopie erstellen, um Original nicht zu verändern
    h_df[:, :Länge] .= parse(Float64, split(teil.Gruppe, "_")[2]) * 0.01 # Neue Spalte hinzufügen, Länge in Meter umrechnen
    append!(PTFE_DF, h_df)  # Zeilen anhängen
end

nrow(PTFE_DF)

filter!([:rMenge, :tges] => (n, t) -> (n <= 100) || (t >= 220), PTFE_DF)
filter!([:tges] => t -> t < 60_000, PTFE_DF)

# Soll_tr-Daten als Achsenabschnitt anlegen
# Kopie + tges=0 in einem Schritt
new_rows = copy(PTFE_DF[PTFE_DF.Isttrsek .> 0, :])
new_rows.tges .= new_rows.Isttrsek
new_rows.rMenge .= 0
# Anhängen
append!(PTFE_DF, new_rows)

vscodedisplay(PTFE_DF)

fig = Figure()
ax = Axis3(fig[1, 1], aspect = (1, 1, 1),
    xlabel = "Azahl",
    ylabel = "Länge [m]",
    zlabel = "tges [s]",
    title = "PTFE-Schläuche: tges vs. nMenge je Länge"
)
bx = Axis(fig[1,2], xlabel = "Anzahl", ylabel = "Länge [m]", title = "Anzahl-Länge")
cx = Axis(fig[2,1], xlabel = "Länge [m]", ylabel = "tges [s]", title = "Länge-tges")
dx = Axis(fig[2,2], xlabel = "Anzahl", ylabel = "tges [s]", title = "Anzahl-tges")

Label(Fig[0,:]," PTFE")
# Gruppe nach Länge und plotte jede Gruppe mit eigenem Label
scatter!(ax, PTFE_DF.rMenge, PTFE_DF.Länge, PTFE_DF.tges, markersize = 5 )
scatter!(bx, PTFE_DF.rMenge, PTFE_DF.Länge, markersize = 5 )
scatter!(cx, PTFE_DF.Länge, PTFE_DF.tges, markersize = 5 )
scatter!(dx, PTFE_DF.rMenge, PTFE_DF.tges, markersize = 5 )

fig

PTFEdict["10-0392m"].df

Regressions_df = DataFrame( y = PTFE_DF.tges, n = PTFE_DF.rMenge, l = PTFE_DF.Länge)

modell_robust = RobustModels.rlm(
    @formula(y ~ n + n&l ),
    Regressions_df,
    RobustModels.MMEstimator{RobustModels.TukeyLoss}();
    σ0 = :mad,
    maxiter = 1000
)

# Gegenchecken wie gut das Modell bei den Top 3 Abschneidet.
DataFrame(StatsAPI.coeftable(modell_robust))

# Extrahiere die Koeffizienten aus dem Modell
coefs = StatsAPI.coef(modell_robust)
intercept = coefs[1]  # Intercept
b_n       = coefs[2]  # n
b_nl      = coefs[3]  # n*l

# Erstelle die Figur
fig = Figure(size = (800, 600))
ax = Axis3(fig[1, 1],
    xlabel = "n (Menge)",
    ylabel = "l (Länge in m)",
    zlabel = "y (Zeit in s)",
    title = "Regressionsfläche: y = $(round(intercept; digits=2)) + $(round(b_n; digits=2))·n + $(round(b_nl; digits=2))·n·l",
)

# 1. Datenpunkte (Scatter)
scatter!(ax, PTFE_DF.rMenge, PTFE_DF.Länge, PTFE_DF.tges,
    markersize = 4,
    color = :blue,
    label = "Datenpunkte",
    transparency = true
)

# 2. Berechne ein Gitter für die Fläche
n_min, n_max = minimum(PTFE_DF.rMenge), maximum(PTFE_DF.rMenge)
l_min, l_max = minimum(PTFE_DF.Länge), maximum(PTFE_DF.Länge)

n_range = range(n_min, n_max, length=50)
l_range = range(l_min, l_max, length=50)

# 3. Berechne die Regressionsfläche
z_matrix = [intercept + b_n*n + b_nl*n*l for n in n_range, l in l_range]

# 4. Plotte die Fläche (Surface)
surface!(ax, n_range, l_range, z_matrix,
    colormap = :viridis,
    transparency = false,
    label = "Regressionsfläche"
)

# 5. Optional: Drahtgitter für bessere 3D-Wahrnehmung
wireframe!(ax, n_range, l_range, z_matrix,
    color = :black,
    linewidth = 0.3,
    transparency = true
)


Gruppen_df = combine(groupby(Aktivitäten_df, :Gruppe), :n => sum)
sort!(Gruppen_df,:n_sum, rev = true)
vscodedisplay(Gruppen_df)


n_max = maximum([teil.n for (key,teil) in Teiledict])

for (key,teil) in Teiledict
    if teil.n == 108 
        println(key)
        break
    end
end

mm_regression!(Teiledict["KK-SK343-75-AA-002"])

ausreißer_flags_setzen!(Teiledict["KK-SK343-75-AA-002"])

vscodedisplay(Teiledict["KK-SK343-75-AA-002"].df)

StatsModels.termvars(@formula(tges ~ rMenge))

test_df, Mpkt, Aus, Auspro = PrepareData!(Teiledict["KK-SK343-75-AA-002"])

theil_sen_regression!(Teiledict["KK-SK343-75-AA-002"])
liste_regressionsmodelle(Teiledict["KK-SK343-75-AA-002"])

fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax,Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"].points.y_hat,Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"].points.residuals)

fig

# Pass 1: λ auf Rohdaten schätzen, Spalte anlegen, Ausreißer identifizieren
λ̂ = yeo_johnson_auf_teil!(Teiledict["KK-SK343-75-AA-002"])
mod = theil_sen_regression!(Teiledict["KK-SK343-75-AA-002"], FORMELN[:yeojohnson])

# Pass 2: λ auf den OK-Punkten NEU schätzen und final fitten
λ̂ = yeo_johnson_auf_teil!(Teiledict["KK-SK343-75-AA-002"]; λ = yeo_johnson_λ(yeo_johnson_inv(mod.points.y, λ̂)))
mod = theil_sen_regression!(Teiledict["KK-SK343-75-AA-002"], FORMELN[:yeojohnson])

# Vorhersagen zurück auf Sekunden:
y_hat_sekunden = yeo_johnson_inv(mod.points.y_hat, λ̂)

koef = StatsAPI.coef(mod.modell)          # [n, m] auf YJ-Skala

yeo_johnson_vorhersage(koef, λ̂, Teiledict["KK-SK343-75-AA-002"].df.rMenge)        # Medianzeit t̂(x) in Sekunden
yeo_johnson_achsenabschnitt(koef, λ̂)     # = t̂(0), das Originalskalen-Pendant zu n
yeo_johnson_steigungen(koef, λ̂, Teiledict["KK-SK343-75-AA-002"].df.rMenge)        # lokale Steigung dy/dx in s/Stück an Stelle x

yeo_johnson_vorhersage(koef, λ̂, 0:0.5:110)


fig = Figure()
cx = Axis(fig[1,1],limits = ((0, 80), nothing))

scatter!(cx, Teiledict["KK-SK343-75-AA-002"].df.rMenge, Teiledict["KK-SK343-75-AA-002"].df.tges)
lines!(cx,(0:0.5:70), yeo_johnson_vorhersage(koef, λ̂, 0:0.5:70))
ax = Axis(fig[2,1], yscale = log, limits = ((0, 80), nothing))
scatter!(ax, Teiledict["KK-SK343-75-AA-002"].df.rMenge, Teiledict["KK-SK343-75-AA-002"].df.tges)
lines!(ax,(0:0.5:70), yeo_johnson_vorhersage(koef, λ̂, 0:0.5:70))
bx = Axis(fig[3,1],limits = ((0, 80), nothing))
density!(bx,Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"].points.x)
linkxaxes!(cx,bx,ax)
fig

yeo_johnson_steigungen(koef, λ̂, [median(Teiledict["KK-SK343-75-AA-002"].df.rMenge)]) 

median(Teiledict["KK-SK343-75-AA-002"].df.rMenge)

mm_regression!(Teiledict["KK-SK343-75-AA-002"])
Teiledict["KK-SK343-75-AA-002"].RegressionsModell["MM-Tukey"].stats
Teiledict["KK-SK343-75-AA-002"].RegressionsModell["MM-Tukey"].points
theil_sen_regression_yj!(Teiledict["KK-SK343-75-AA-002"])


# Pass 1: λ auf Rohdaten schätzen, Spalte anlegen, Ausreißer identifizieren
λ̂ = yeo_johnson_auf_teil!(Teiledict["KK-SK343-75-AA-002"])
mod = mm_regression!(Teiledict["KK-SK343-75-AA-002"], FORMELN[:yeojohnson])

# Pass 2: λ auf den OK-Punkten NEU schätzen und final fitten
λ̂ = yeo_johnson_auf_teil!(Teiledict["KK-SK343-75-AA-002"]; λ = yeo_johnson_λ(yeo_johnson_inv(mod.points.y, λ̂)))
mod = mm_regression!(Teiledict["KK-SK343-75-AA-002"], FORMELN[:yeojohnson])

liste_regressionsmodelle(Teiledict["KK-SK343-75-AA-002"])
diagnostik!(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"])
bewerte_diagnostik!(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"])
vscodedisplay(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["Theil-Sen"].Diagnostik)

mm_regression!(Teiledict["KK-SK343-75-AA-002"])
liste_regressionsmodelle(Teiledict["KK-SK343-75-AA-002"])

diagnostik!(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["MM-Tukey"])
bewerte_diagnostik!(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["MM-Tukey"])
vscodedisplay(Teiledict["KK-SK343-75-AA-002"].RegressionsModell["MM-Tukey"].Diagnostik)

# BeiPackDaten laden
BP_df = CSV.read("BeiPackTeile.CSV", DataFrame)
# leere Dicter stellen
BPdict = Dict{String,Dict}()
# Dictonary der Beipackteile mit den Materialien und deren Anzahl füllen
for i in 1:nrow(BP_df)
    BeiPack = Dict{String,Int64}()
    BPstr = split(BP_df[i, :Material], ",")
    for bp in BPstr
        bp_split = split(bp, "x")
        BeiPack[bp_split[2]] = parse(Int64, bp_split[1])
    end
    BPdict[BP_df[i,1]] = BeiPack
end

# Dataframe mit allen Materialien erstellen, die in den Beipackteilen vorkommen
# Lister der Materilien erstllen
c_unique = Dict{String,Bool}()
for (key,Material) in BPdict
    for (mkey, BeiPack) in Material
        c_unique[mkey] = true
    end
end
# Materialien alphabetisch sortieren
c_unique_sorted = c_unique |> keys |> collect |> sort
# DataFrame für die Beipackteile erstellen
BP_df_long = DataFrame(BP = String.(BP_df[:, 1]))
for material in c_unique_sorted
    BP_df_long[!, Symbol(material)] = zeros(Int, nrow(BP_df_long))
end

# DataFrame mit Materialmengen je Beipackteil füllen
bp_zeile = Dict(bp => i for (i, bp) in enumerate(BP_df_long.BP))
for (bp, materialien) in BPdict
    i = bp_zeile[bp]
    for (material, anzahl) in materialien
        BP_df_long[i, Symbol(material)] = anzahl
    end
end

# Anzahl der Materialien je Beipackteil ermitteln
BP_df_long[!, :n] = [length(BPdict[bp]) for bp in BP_df_long.BP]

vscodedisplay(BP_df_long)
CSV.write("BeiPackTeile_long.CSV", BP_df_long)

BPTeiledict = Dict{String,Teil}()

for (key,teil) in Teiledict
    if startswith(teil.name, "BP")
        BPTeiledict[key] = teil
    end
end

bp_dfs = DataFrame[]

for (bp, teil) in BPTeiledict
    df = copy(teil.df)
    df[!, :BP] = fill(bp, nrow(df))
    push!(bp_dfs, df)
end

BP_records = vcat(bp_dfs...; cols = :union)
BP_df_gesamt = leftjoin(BP_records, BP_df_long; on = :BP)
BP_df_gesamt[!, :BP] = categorical(BP_df_gesamt.BP)

modell = MixedModels.fit(
    MixedModel,
    @formula(tges ~ 1 + n + (1 | BP)),
    BP_df_gesamt
)