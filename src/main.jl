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
    "Random",
    "Pluto",
    "MixedModels",
    "Turing",
    "Distributions",
    "MCMCChains",
    "SpecialFunctions"
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
using Random                # Funktionserweiterung für den Zufallsgenerator
using LinearAlgebra         # Matrixoperationen und anders mathematisches gedöns
#using Pluto
using Dates
#using MixedModels           # MixedModels-Ansatz mit max.-liklyhood-Methodig, nicht-robust
using Turing                 # Bayes-Infrastruktur für das hierarchische robuste Modell (Weg 2)
using Distributions          # Verteilungen für Priors und Student-t-Fehler
using MCMCChains             # Posterior-Ketten und Diagnostik (R-hat, ESS)
using SpecialFunctions       # lgamma für die Student-t-Logdichte


include("Konstructs.jl")                # Kontainervariablen und deren Hilfsfunktionen
include("Diagrams.jl")                  # Funktionen zur Datendarstellung
include("TideyUpData.jl")               # Hilfsfunktionen zum bearbeiten von Datensätzen
include("YeoJohnson_Transformation.jl") # Yeo-Johnson-Transformation: λ-Schätzung, Vorwärts/Rücktransformation
include("Regressionsmethoden.jl")       # Funktionen für die Regressionenmethoden wie Theil Sen, MM, ...
include("Regressionsvalidation.jl")     # Funktion zum Validieren des Regressionsmodells
include("Ausreißer_Flags.jl")           # Flag-System zur Ausreißerklassifikation
include("Montecarlo_Sim.jl")            # Montecarlo-Simulation zum Abschätzen des Pausenzeitfehlers
include("whittaker_eiler_smoother.jl")  # Smoother/Filter ohne Verzögerung im Output
include("Hierarchische_Regression.jl")  # Bayes-hierarchisches robustes Modell (Weg 2, Turing)
include("df_Hilfsfunktionen.jl")        # Hilfsfunktionen zum Erkunden und Bearbeiten von DataFrames 


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
vscodedisplay(df)

Teiledict = erstelle_teile(df)
@save "Teiledict.jld2" Teiledict
@load "Teiledict.jld2" Teiledict

# Speichern der Eintragsverteilung als SVG-Datei
CairoMakie.activate!()
fig = show_einträge(Teiledict)
save("eintragsverteilung.svg", fig)

# BDE-Daten laden
df2 = CSV.read("PPA-BDE02_AbRMNr43000.csv",DataFrame)
# ab RMNr 43000 alle Daten verwenden
filter!(:RMNr => nr -> nr > 42999, df2)
# Spalten konvertieren
df2.Teil = String.(df2.Teil)
df2.MA = String.(df2.MA)
# nach AOI filtern
AOI_df = filter(:Teil => startswith("OI-"), df2)
# Platine extrahieren
AOI_df.Platine = String.([s[2] for s in split.(AOI_df.Teil,"-")])
# Ein Dictonary aus Teilen des BDE-Dataframes erstellen. 
df_gruppe_anlegen!(AOI_df,:Platine,n_min = 10)

# tges-Spalte hinzufügen
AOI_df.tges = AOI_df.Zeit
# Gruppieren ohne Sonstiges

AOI_group = groupby(filter(:Gruppe => t -> t != "Sonstiges",AOI_df),:Gruppe)

# Kurz sichten
Plot_Gruppe(AOI_group, :rMenge, :Zeit)

# Ausreißerentfernen
AOI_Teile_group = erstelle_teile_aus_gruppe(AOI_group)
# Regression über jede Gruppe
for (key, teil) in AOI_Teile_group
    theil_sen_regression_yj!(teil)
    diagnostik!(teil.RegressionsModell["Theil-Sen"])
    bewerte_diagnostik!(teil.RegressionsModell["Theil-Sen"])
end

# Kurz sichten
f = Plot_Gruppe(AOI_Teile_group, :rMenge, :Zeit)
diag_df = collect_diagnistic(AOI_Teile_group,"Theil-Sen")
AOI_Teile_group["VAG01e"].RegressionsModell["Theil-Sen"].Diagnostik

# Ausreißer entfernen
AOI_Teile_df = filter(:Ausreißer => a -> a != "Ausreißer", df_aus_Teiledict(AOI_Teile_group))

# Metadaten erstellen, Zum Vergleich der Ausreißer
AOI_Teile_Meta_mod = TheilSenModell(FORMELN[:linear],AOI_Teile_df)


# MixedModel-Ansatz
using MixedModels
AOI_lmm_erg = lmm_regression!(AOI_Teile_group; quelle = :Zeit)

vscodedisplay(AOI_lmm_erg.fix)       # = dein DataFrame(coeftable(lmm))
vscodedisplay(AOI_lmm_erg.gruppen)    # = dein AOI_lmm (inkl. Zerlegung c/rng/all)
liste_regressionsmodelle(AOI_Teile_group["VAG01e"])   # enthält jetzt "MixedModel"
Plot_Gruppe(AOI_Teile_group, :rMenge, :Zeit; Regression = "MixedModel")

# Bayers
# Aufbauend auf Theil Sen die zur Ausreißerbereinigung dient. 
"""
----
# Weg 2: Hierarchisches Bayes-Modell (Turing) — partielles Pooling auf ALLEN Messpunkten
----
Anders als Option D (Stufen 1-3) nutzt dieses Modell jede Messung: Gruppen
mit wenig Daten schrumpfen Richtung Gesamtmuster statt rauszufliegen, und die
Student-t-Fehler (ν geschätzt) gewichten Ausreißer automatisch herunter.
Laufzeit: Der erste Aufruf kompiliert (Minuten); zum Testen N_samples/ketten
reduzieren, für die Auswertung 500 × 4.
"""
# Datengrundlage für das Bayes-Modell: die per Theil-Sen+YJ identifizierten
# harten Ausreißer sind ausgeschlossen (Stufe-1-Flags aus AOI_teile), "Verdacht"
# bleibt drin. Damit widmet sich das Modell dem Pooling statt der Schadens-
# begrenzung von Datenfehlern — ν sollte danach vom Anschlag (≈2) aufsteigen.
# Gruppen, deren Stufe-1-Regression fehlschlug (keine :Ausreißer-Spalte),
# fließen ungefiltert ein.
# "Sonstiges" bleibt draußen (gemischte Restgruppe). Mit partiellem Pooling
# könnte künftig sogar die >=10-Einträge-Regel entfallen: kleine Platinen
# bekommen dann einfach stark geschrumpfte Koeffizienten.

AOI_bayes = robust_bayes_lmm(AOI_Teile_df, :Zeit, :rMenge, :Gruppe;
                             N_samples = 1000, ketten = 4)

# Kette sichern — Achtung Syntax: @save nimmt Variablennamen bzw. name = ausdruck,
# keine Feldzugriffe wie "AOI_bayes.kette" (daher kam der "Invalid field syntax"-Fehler).
@save "AOI_bayes_kette.jld2" kette = AOI_bayes.kette
# Falls das Chain-Objekt nicht sauber serialisiert (komplexe Typen zwischen
# Paketversionen), die Draws als DataFrame sichern; robust_bayes_lmm akzeptiert
# den DataFrame ebenfalls als `kette`:
# @save "AOI_bayes_kette.jld2" kette = DataFrame(AOI_bayes.kette)
# Wiederverwendung später:
@load "AOI_bayes_kette.jld2" kette
AOI_bayes = robust_bayes_lmm(AOI_Teile_df, :Zeit, :rMenge, :Gruppe; kette = kette)

vscodedisplay(AOI_bayes.fix)      # globale Koeffizienten, σ, τ0, τ1, ν (Originalskala)
vscodedisplay(AOI_bayes.gruppe)  # partiell gepoolte Koeffizienten je Platine
Plot_Bayes_Gruppen(AOI_Teile_df; bayes = AOI_bayes)


# Vergleich Stufe 1 (Theil-Sen, isoliert) gegen Weg 2 (gepoolt):
# Punkte unterhalb/oberhalb der Diagonalen = geschrumpfte Gruppen.
vergleich = innerjoin(
    select(AOI_ergebnisse, :Gruppe, :m => :m_theilsen, :n_OK, :Ausschluss),
    select(AOI_bayes.gruppe, :Gruppe, :m => :m_bayes),
    on = :Gruppe)
vscodedisplay(vergleich)

fig = Figure(size = (1100, 500))
ax1 = Axis(fig[1, 1],
    xlabel = "Theil-Sen m [s/Stück] (Stufe 1, je Gruppe isoliert)",
    ylabel = "Bayes m [s/Stück] (partiell gepoolt)",
    title = "Schrumpfung: Gruppen mit wenig Daten wandern Richtung Pool")
pt = scatter!(ax1, vergleich.m_theilsen, vergleich.m_bayes,
              color = vergleich.n_OK, colormap = :viridis, markersize = 14)
ablines!(ax1, 0, 1, color = :gray, linestyle = :dash)
Colorbar(fig[2, 1], pt, label = "n_OK der Gruppe (Stufe 1)", vertical = false, flipaxis = false)

ax2 = Axis(fig[1, 2], xlabel = "rMenge", ylabel = "Zeit [s]",
           title = "Bayes-Geraden je Platine (partiell gepoolt)")
scatter!(ax2, AOI_fit_df.rMenge, AOI_fit_df.Zeit, color = (:gray, 0.3), markersize = 4)
x_lin = range(minimum(AOI_fit_df.rMenge), maximum(AOI_fit_df.rMenge), length = 50)
for r in eachrow(AOI_bayes.gruppe)
    lines!(ax2, x_lin, r.a .+ r.m .* x_lin, linewidth = 2)
end
fig

# Interaktiver Gruppenplot mit Toggles: Punktwolke + Bayes-Gerade je Platine
Plot_Bayes_Gruppen(AOI_fit_df; bayes = AOI_bayes)

AOI_teile["VAG01e"].df

Plot_Gruppe(AOI_teile,:rMenge, :Zeit)

liste_regressionsmodelle(AOI_teile["Sonstiges"])

"""
----
# Weg 2: Validierung des Bayes-Modells
----
Drei Ebenen (Funktionen in Hierarchische_Regression.jl / Diagrams.jl):
1. Sampler-Diagnostik: R-hat/ESS über ALLE Parameter inkl. Zufallseffekte
   (die fix-Tabelle prüft nur α, γ, σ, τ0, τ1, ν).
2. Residuen-Diagnostik: Trichter-Check (Heteroskedastizität!), QQ gegen t(ν).
3. Abdeckung je Gruppe: liegen ~95 % der Punkte im 95 %-Band der t(ν)-Fehler?
"""
# 1. MCMC-Diagnostik (Achtung: benötigt die Chain-Objekt-Kette, nicht den
#    per DataFrame gesicherten Draw-Table)
vscodedisplay(bayes_mcmc_diagnostik(AOI_bayes))

# 2. Residuen je Messpunkt + Panel-Plot (mitnehmen: Spalten zum Einfärben)
AOI_residuen = bayes_residuen(AOI_Teile_df, :Zeit, :rMenge, :Gruppe, AOI_bayes;
                              mitnehmen = [:MA, :Arbeiter, :RMNr])
Plot_Bayes_Residuen(AOI_residuen; ν = AOI_bayes.ν)

# Stränge in Residuum vs. Vorhersage aufklären: identifizieren die Stränge
# sich mit Gruppe (links) oder MA/Arbeitsplatz (rechts)?
# Achtung: color = Vector{String} versteht Makie als Farbnamen, nicht als
# Kategorien — daher je Kategorie ein scatter! mit Farbe aus resample_cmap
# (liefert zugleich echte Legenden-Einträge).
fig = Figure(size = (1100, 500))
ax_g = Axis(fig[1, 1], xlabel = "Vorhersage [s]", ylabel = "Residuum [s]",
            title = "Residuum vs. Vorhersage — Farbe: Gruppe")
gruppen = unique(AOI_residuen.Gruppe)
farben_g = resample_cmap(:tab20, length(gruppen))
for (i, g) in enumerate(gruppen)
    sel = AOI_residuen.Gruppe .== g
    scatter!(ax_g, AOI_residuen.vorhersage[sel], AOI_residuen.residuum[sel],
            color = farben_g[i], markersize = 6, label = string(g))
end
hlines!(ax_g, 0, color = :red, linestyle = :dash)
axislegend(ax_g, position = :rt, nbanks = 2)

ax_m = Axis(fig[1, 2], xlabel = "Vorhersage [s]", ylabel = "Residuum [s]",
            title = "Residuum vs. Vorhersage — Farbe: MA")
mas = unique(AOI_residuen.MA)
farben_m = resample_cmap(:tab10, length(mas))
for (i, m) in enumerate(mas)
    sel = AOI_residuen.MA .== m
    scatter!(ax_m, AOI_residuen.vorhersage[sel], AOI_residuen.residuum[sel],
            color = farben_m[i], markersize = 6, label = string(m))
end
hlines!(ax_m, 0, color = :red, linestyle = :dash)
axislegend(ax_m, position = :rt)
fig

# 3. Abdeckung und Güte je Gruppe
vscodedisplay(bayes_abdeckung(AOI_residuen; ν = AOI_bayes.ν))
vscodedisplay(AOI_Teile_group["VAG01e"].df)

CairoMakie.activate!()
f = Plot_Teil(AOI_Teile_group["VAG01e"])
save("test-vag01e.pdf",f) # Kleinste Datei

liste_regressionsmodelle(AOI_Teile_group["VAG01e"])

AOI_Teile_group["VAG01e"].df

loo_mad_diagnostics(AOI_Teile_group["VAG01e"].df,:TheilSen)

"""
----
# Weg 2: Bayes-Regression als Regressionsmodell je Gruppe speichern
----
bayes_regression! (Hierarchische_Regression.jl) bündelt den kompletten
Workflow: Theil-Sen-YJ-Check je Teil (wird nachgeholt, falls "Theil-Sen YJ"
fehlt), gepoolter Fit via robust_bayes_lmm, Ablage des Ergebnisses je
Gruppe als RegressionsModell "Bayes" — analog zu lmm_regression!.
"Sonstiges" bleibt vom Fit ausgeschlossen und erhält nur die statischen
Effekte plus eine Verwerfen-Diagnose (zu wenige Einträge).
"""

# Variante B: bestehendes Teile-Dict. Theil-Sen YJ ist hier bereits
# durchgeführt — der Check in bayes_regression! erkennt das am Eintrag
# "Theil-Sen YJ" und fitet nicht neu. kette = kette nutzt die oben
# gesicherte Posterior-Kette (gleicher Datenstand!), kein neues Sampling.
AOI_bayes_erg = bayes_regression!(AOI_Teile_group; quelle = :Zeit, kette = kette)

vscodedisplay(AOI_bayes_erg.gruppe)                    # gepoolte Koeffizienten je Gruppe
liste_regressionsmodelle(AOI_Teile_group["VAG01e"])    # enthält jetzt "Bayes"
Plot_Gruppe(AOI_Teile_group, :rMenge, :Zeit; Regression = "Bayes")

# Diagnostik je Gruppe (Abdeckung) — sammelbar wie bei Theil-Sen:
diag_df_bayes = collect_diagnistic(AOI_Teile_group, "Bayes")
vscodedisplay(diag_df_bayes)

# Variante A: aus dem Roh-DataFrame — legt die Gruppen (inkl. "Sonstiges"
# für Kleingruppen < n_min Einträge) selbst an, wie das
# df_gruppe_anlegen!-Beispiel oben, und fitet anschließend:
# erg = bayes_regression!(AOI_df, :Platine; quelle = :Zeit)
# erg.teile["Sonstiges"].RegressionsModell["Bayes"].Diagnostik
# # → Stufe "Verwerfen", Begründung: zu wenige Einträge (nur statische Effekte)

AOI_test_df, AOI_dict = bayes_regression!(AOI_df,:Platine; quelle = :Zeit)

AOI_test_df |> typeof
AOI_dict |> typeof

AOI_dict
AOI_group = groupby(AOI_df,:Gruppe)

AOI_testdict = erstelle_teile_aus_gruppe(AOI_group)

liste_regressionsmodelle(AOI_testdict["VAG01e"])

test = bayes_regression!(AOI_testdict)

liste_regressionsmodelle(AOI_testdict["VAG01e"])

@save "AOI_Gruppen_Bayes.jld2" AOI_testdict

keys(AOI_testdict)
AOI_testdict["VAG01e"].RegressionsModell["Bayes"].Diagnostik |> vscodedisplay


AOI_residuen = bayes_residuen(AOI_Teile_df, :Zeit, :rMenge, :Gruppe, AOI_bayes;
                              mitnehmen = [:MA, :Arbeiter, :RMNr])

vscodedisplay(bayes_abdeckung(AOI_residuen; ν = AOI_bayes.ν))

Diagnosen_sammeln(AOI_testdict)

test = spalte_gruppe_in_teile(AOI_testdict; meta = Teiledict)
@save "AOI_Gruppen_gespalten_Bayes.jld2" test

test["OI-VAG01e-A-01"].RegressionsModell["Bayes"].Diagnostik
test["OI-VAG01e-A-01"].Gruppe

function Diagnostik_sammeln(teiledict::Dict{String,Teil},Regression::String="Bayes")
    dfs = Vector{DataFrame}(undef,length(teiledict))

    for (i,(key,teil)) in enumerate(teiledict)
        reg = teil.RegressionsModell
        if haskey(reg,Regression)
            reg = reg[Regression]
            m = round(reg.stats.m[1], digits = 2)
            m_lo = round(reg.stats.m_lo[1], digits = 2)
            m_hi = round(reg.stats.m_hi[1], digits = 2)
            n = round(reg.stats.n[1]/60, digits = 2)
            n_lo = round(reg.stats.a_lo[1]/60, digits = 2) # inkonstistenz Intercept ist bei bayes als a bezeichnet statt als n
            n_hi = round(reg.stats.a_hi[1]/60, digits = 2)
            Einstufung = reg.Diagnostik.Stufe[1]
        else
            trow(ArgumentError("Die Regression : $(Regression) ist in $(teil.name) nicht vorhanden."))
        end
        dfs[i] = DataFrame( :Gruppe => teil.Gruppe,
                            :Teil => teil.name,
                            :Einschätzung => Einstufung,
                            :soll_tr_min => round(teil.Soll_tr / 60, digits = 2),
                            :tr_low => n_lo,
                            :tr_mid => n,
                            :tr_hi => n_hi,
                            :real_tr => round(n/(teil.Soll_tr/60), digits = 4),
                            :soll_te_sek => round(teil.Soll_te, digits = 4),
                            :te_low => m_lo,
                            :te_mid => m,
                            :te_hi => m_hi,
                            :real_te => round(m/teil.Soll_te, digits = 4)
                            )
    end
    return vcat(dfs...)
end

Result = Diagnostik_sammeln(test)

sort!(Result, [:Gruppe, order(:Teil)]) # Sortieren mit Numerbeachtung

using XLSX

save("test.xlsx", Result)

Plot_Teil(test["OI-VAG01e-A-01"])
test["OI-VAG01e-A-01"].Bezeichnung

speichere_result_excel(Result, "Result.xlsx")